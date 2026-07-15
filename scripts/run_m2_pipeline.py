"""Module 2 complete pipeline: GT tracking → field mapping → baselines → ConvLSTM.

Runs the full GT-based temporal chain on WildTrack annotations:
  1. Load annotations, build trajectories
  2. Evaluate tracking (NN + Kalman) on GT detections
  3. Build occupancy/velocity fields for all frames
  4. Evaluate non-learning baselines (persistence, constant velocity, advection, oracle)
  5. Train ConvLSTM and evaluate vs baselines
  6. Output comprehensive results JSON

Usage:
    PYTHONPATH=src python3 scripts/run_m2_pipeline.py \
        --annotations_dir wildtrack/annotations_positions \
        --output_dir outputs/m2_pipeline \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

np_rng = np.random.RandomState(0)


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


def run_tracking_eval(annotations_dir: Path, output_dir: Path) -> dict:
    """Run NN and Kalman trackers on GT detections, evaluate MOTA/IDF1."""
    from temporal.annotation_reader import load_all_annotations, build_trajectories
    from temporal.tracker_nn import NearestNeighborTracker
    from temporal.tracker_kalman import KalmanHungarianTracker
    from temporal.tracking_metrics import evaluate_tracking
    from temporal.time_utils import get_split_range

    val_start, val_end = get_split_range("val")
    frames = load_all_annotations(annotations_dir, frame_start=val_start, max_frames=val_end - val_start)

    gt_frames = []
    det_frames = []
    for fi, frame_dets in enumerate(frames):
        positions = np.array([[d.world_x_m, d.world_y_m] for d in frame_dets]) if frame_dets else np.empty((0, 2))
        ids = np.array([d.person_id for d in frame_dets]) if frame_dets else np.array([], dtype=np.int64)
        gt_frames.append({"positions": positions, "ids": ids})
        det_frames.append(positions)

    results = {}
    for tracker_name, tracker_cls, params in [
        ("nn", NearestNeighborTracker, {"dist_gate": 1.0, "max_age": 2, "min_hits": 2}),
        ("kalman", KalmanHungarianTracker, {"dist_gate": 1.0, "max_age": 2, "min_hits": 2}),
    ]:
        tracker = tracker_cls(**params)
        pred_frames = []
        for fi, dets in enumerate(det_frames):
            out = tracker.update(dets, frame_index=val_start + fi)
            pred_pos = np.array([[t.world_x_m, t.world_y_m] for t in out.active_tracks]) if out.active_tracks else np.empty((0, 2))
            pred_ids = np.array([t.track_id for t in out.active_tracks]) if out.active_tracks else np.array([], dtype=np.int64)
            pred_frames.append({"positions": pred_pos, "ids": pred_ids})

        metrics = evaluate_tracking(gt_frames, pred_frames, dist_thr=0.5)
        results[tracker_name] = {
            "mota": metrics.mota, "idf1": metrics.idf1,
            "id_switches": metrics.id_switches, "fragmentations": metrics.fragmentations,
            "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn,
        }
        print(f"  [{tracker_name}] MOTA={metrics.mota:.4f} IDF1={metrics.idf1:.4f} "
              f"IDSW={metrics.id_switches} Frag={metrics.fragmentations}")

    return results


def build_all_frame_fields(annotations_dir: Path, start: int, n_frames: int, sigma_m: float) -> np.ndarray:
    """Build field tensors for a range of frames."""
    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities
    from temporal.field_builder import build_all_fields
    from temporal.time_utils import DT

    frames = load_all_annotations(annotations_dir, frame_start=start, max_frames=n_frames)
    trajectories = build_trajectories(frames)

    person_velocities: dict[int, dict[int, np.ndarray]] = {}
    for pid, traj in trajectories.items():
        vel = compute_velocities(traj, dt=DT)
        for i, det in enumerate(traj.detections):
            if det.frame_index not in person_velocities:
                person_velocities[det.frame_index] = {}
            person_velocities[det.frame_index][pid] = vel[i]

    all_fields = np.zeros((n_frames, 5, 120, 360), dtype=np.float32)
    for fi in range(n_frames):
        abs_fi = start + fi
        dets = frames[fi]
        if not dets:
            continue
        positions = np.array([[d.world_x_m, d.world_y_m] for d in dets])
        velocities = np.zeros((len(dets), 2), dtype=np.float64)
        for j, d in enumerate(dets):
            vel_map = person_velocities.get(d.frame_index, {})
            if d.person_id in vel_map:
                velocities[j] = vel_map[d.person_id]
        all_fields[fi] = build_all_fields(positions, velocities, sigma_m=sigma_m)

    return all_fields


def eval_baselines(annotations_dir: Path, sigma_m: float) -> dict:
    """Evaluate non-learning baselines on validation split."""
    from temporal.baselines import (
        predict_persistence, predict_constant_velocity,
        predict_field_advection, predict_oracle,
    )
    from temporal.field_metrics import compute_occupancy_auprc, compute_velocity_epe
    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities
    from temporal.time_utils import get_split_range, DT

    val_start, val_end = get_split_range("val")
    n_val = val_end - val_start
    history_len = 4
    future_len = 4

    fields = build_all_frame_fields(annotations_dir, val_start, n_val, sigma_m)
    frames = load_all_annotations(annotations_dir, frame_start=val_start, max_frames=n_val)
    trajectories = build_trajectories(frames)

    person_velocities: dict[int, dict[int, np.ndarray]] = {}
    for pid, traj in trajectories.items():
        vel = compute_velocities(traj, dt=DT)
        for i, det in enumerate(traj.detections):
            if det.frame_index not in person_velocities:
                person_velocities[det.frame_index] = {}
            person_velocities[det.frame_index][pid] = vel[i]

    results = {}
    n_windows = n_val - history_len - future_len + 1

    for baseline_name in ["persistence", "constant_velocity", "advection", "oracle"]:
        auprc_list = []
        epe_list = []

        for wi in range(max(n_windows, 1)):
            t_now = history_len + wi - 1
            if t_now + future_len > n_val:
                break

            current_occ = fields[t_now, 0]
            current_vx = fields[t_now, 1]
            current_vy = fields[t_now, 2]

            gt_future = [fields[t_now + 1 + s, 0] for s in range(future_len)]
            gt_future_vx = [fields[t_now + 1 + s, 1] for s in range(future_len)]
            gt_future_vy = [fields[t_now + 1 + s, 2] for s in range(future_len)]

            if baseline_name == "persistence":
                preds = predict_persistence(current_occ, future_len)
            elif baseline_name == "constant_velocity":
                dets = frames[t_now]
                if dets:
                    pos = np.array([[d.world_x_m, d.world_y_m] for d in dets])
                    vels = np.zeros((len(dets), 2), dtype=np.float64)
                    for j, d in enumerate(dets):
                        vm = person_velocities.get(d.frame_index, {})
                        if d.person_id in vm:
                            vels[j] = vm[d.person_id]
                    preds = predict_constant_velocity(pos, vels, future_len, dt=DT, sigma_m=sigma_m)
                else:
                    preds = [np.zeros((120, 360), dtype=np.float32)] * future_len
            elif baseline_name == "advection":
                preds = predict_field_advection(current_occ, current_vx, current_vy, future_len)
            elif baseline_name == "oracle":
                future_positions = []
                for s in range(future_len):
                    fi = t_now + 1 + s
                    if fi < n_val:
                        dets_f = frames[fi]
                        pos_f = np.array([[d.world_x_m, d.world_y_m] for d in dets_f]) if dets_f else np.empty((0, 2))
                    else:
                        pos_f = np.empty((0, 2))
                    future_positions.append(pos_f)
                preds = predict_oracle(future_positions, sigma_m=sigma_m)

            for s in range(future_len):
                auprc = compute_occupancy_auprc(preds[s], gt_future[s])
                auprc_list.append(auprc)

        results[baseline_name] = {
            "occ_auprc_mean": float(np.mean(auprc_list)) if auprc_list else 0.0,
            "occ_auprc_std": float(np.std(auprc_list)) if auprc_list else 0.0,
            "n_evaluations": len(auprc_list),
        }
        print(f"  [{baseline_name}] AUPRC={results[baseline_name]['occ_auprc_mean']:.4f} "
              f"±{results[baseline_name]['occ_auprc_std']:.4f} (n={len(auprc_list)})")

    return results


def train_and_eval_convlstm(annotations_dir: Path, output_dir: Path,
                            device_str: str, seed: int, sigma_m: float) -> dict:
    """Train ConvLSTM and evaluate on validation split."""
    import torch
    from torch.utils.data import DataLoader
    from temporal.convlstm import SpatioTemporalPredictor
    from temporal.temporal_loss import CombinedTemporalLoss
    from temporal.temporal_dataset import FieldSequenceDataset
    from temporal.temporal_trainer import TemporalTrainer, set_seed
    from temporal.field_metrics import compute_occupancy_auprc

    set_seed(seed)
    device = torch.device(device_str)

    print(f"  Loading datasets (seed={seed})...")
    train_ds = FieldSequenceDataset(annotations_dir, split="train",
                                    history_len=4, future_len=4, sigma_m=sigma_m)
    val_ds = FieldSequenceDataset(annotations_dir, split="val",
                                  history_len=4, future_len=4, sigma_m=sigma_m)
    print(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            pin_memory=device.type == "cuda")

    model = SpatioTemporalPredictor(
        in_channels=5, hidden_channels=32, kernel_size=3,
        n_encoder_layers=2, n_future=4, out_channels=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    criterion = CombinedTemporalLoss(lambda_vel=0.5, lambda_trace=0.1, ablation="full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    run_dir = output_dir / f"convlstm_seed{seed}"
    trainer = TemporalTrainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, device=device, output_dir=run_dir, patience=10,
    )

    t_start = time.time()
    for epoch in range(100):
        train_metrics = trainer.train_epoch(train_loader, epoch, log_every=50)
        val_metrics = trainer.validate(val_loader)

        if val_metrics["auprc"] > trainer.best_auprc or trainer.best_auprc < 0:
            trainer.save_checkpoint(epoch, val_metrics, best=True)
        trainer.save_checkpoint(epoch, val_metrics)

        if epoch % 10 == 0 or epoch < 3:
            print(f"  [epoch {epoch}] train={train_metrics['total']:.6f} "
                  f"val={val_metrics['total']:.6f} auprc={val_metrics['auprc']:.4f}")

        if trainer.check_early_stop(val_metrics["auprc"]):
            print(f"  [STOP] Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"  Training done in {elapsed:.0f}s, best AUPRC={trainer.best_auprc:.4f}")

    return {
        "best_auprc": trainer.best_auprc,
        "final_epoch": epoch,
        "elapsed_s": elapsed,
        "n_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Module 2 complete pipeline")
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/m2_pipeline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sigma_m", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── 1. Tracking evaluation ──
    banner("1/4  Tracking Evaluation (GT detections, val split)")
    all_results["tracking"] = run_tracking_eval(ann_dir, output_dir)

    # ── 2. Field construction stats ──
    banner("2/4  Field Construction Stats")
    from temporal.time_utils import get_split_range
    for split in ["train", "val", "test"]:
        s, e = get_split_range(split)
        fields = build_all_frame_fields(ann_dir, s, e - s, args.sigma_m)
        occ_mean = fields[:, 0].mean()
        occ_max = fields[:, 0].max()
        vel_mean = np.sqrt(fields[:, 1] ** 2 + fields[:, 2] ** 2).mean()
        print(f"  [{split}] frames={e - s}, occ_mean={occ_mean:.6f}, "
              f"occ_max={occ_max:.4f}, vel_mean={vel_mean:.4f} m/s")
        all_results[f"fields_{split}"] = {
            "n_frames": e - s,
            "occ_mean": float(occ_mean),
            "occ_max": float(occ_max),
            "vel_mean": float(vel_mean),
        }

    # ── 3. Non-learning baselines ──
    banner("3/4  Non-Learning Baselines (val split)")
    all_results["baselines"] = eval_baselines(ann_dir, args.sigma_m)

    # ── 4. ConvLSTM training ──
    if not args.skip_training:
        banner("4/4  ConvLSTM Training (full ablation, seed={})".format(args.seed))
        all_results["convlstm"] = train_and_eval_convlstm(
            ann_dir, output_dir, args.device, args.seed, args.sigma_m
        )
    else:
        print("\n[SKIP] ConvLSTM training skipped (--skip_training)")
        all_results["convlstm"] = {"skipped": True}

    # ── Summary ──
    banner("RESULTS SUMMARY")
    results_path = output_dir / "m2_pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if "baselines" in all_results:
        print("\n  Non-learning baselines (val AUPRC):")
        for name, m in all_results["baselines"].items():
            print(f"    {name:20s}: {m['occ_auprc_mean']:.4f} ±{m['occ_auprc_std']:.4f}")

    if "convlstm" in all_results and "best_auprc" in all_results["convlstm"]:
        print(f"\n  ConvLSTM (best val AUPRC): {all_results['convlstm']['best_auprc']:.4f}")

    if "tracking" in all_results:
        print("\n  Tracking (val split):")
        for name, m in all_results["tracking"].items():
            print(f"    {name:10s}: MOTA={m['mota']:.4f} IDF1={m['idf1']:.4f}")

    print("\n[DONE] Pipeline complete", flush=True)


if __name__ == "__main__":
    main()
