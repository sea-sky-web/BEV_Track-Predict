"""Module 2 complete pipeline: three-level evaluation framework.

Level 1 (GT):       GT positions + GT identity → pure prediction capability
Level 2 (Det+GT):   Detector positions + GT association → detection error impact
Level 3 (Det+Trk):  Detector positions + Tracker → full end-to-end

Usage:
    # GT-only (no checkpoint needed):
    PYTHONPATH=src python3 scripts/run_m2_pipeline.py \
        --annotations_dir wildtrack/annotations_positions \
        --output_dir outputs/m2_pipeline \
        --device cuda

    # Full three-level (needs detector JSONL):
    PYTHONPATH=src python3 scripts/run_m2_pipeline.py \
        --annotations_dir wildtrack/annotations_positions \
        --detections_jsonl outputs/frozen_detector/detections.jsonl \
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


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


# ── Tracking evaluation ──

def _run_tracker_on_frames(tracker, positions_per_frame, frame_offset):
    """Run a tracker on per-frame positions, return pred_frames for metric eval."""
    pred_frames = []
    for fi, pos in enumerate(positions_per_frame):
        out = tracker.update(pos, frame_index=frame_offset + fi)
        if out.active_tracks:
            pred_pos = np.array([[t.world_x_m, t.world_y_m] for t in out.active_tracks])
            pred_ids = np.array([t.track_id for t in out.active_tracks], dtype=np.int64)
        else:
            pred_pos = np.empty((0, 2))
            pred_ids = np.array([], dtype=np.int64)
        pred_frames.append({"positions": pred_pos, "ids": pred_ids})
    return pred_frames


def eval_tracking(gt_frames_data, input_positions, frame_offset, level_name):
    """Evaluate NN + Kalman tracking on given input positions."""
    from temporal.tracker_nn import NearestNeighborTracker
    from temporal.tracker_kalman import KalmanHungarianTracker
    from temporal.tracking_metrics import evaluate_tracking

    results = {}
    for tracker_name, tracker_cls, params in [
        ("nn", NearestNeighborTracker, {"dist_gate": 1.0, "max_age": 2, "min_hits": 2}),
        ("kalman", KalmanHungarianTracker, {"dist_gate": 1.0, "max_age": 2, "min_hits": 2}),
    ]:
        tracker = tracker_cls(**params)
        pred_frames = _run_tracker_on_frames(tracker, input_positions, frame_offset)
        metrics = evaluate_tracking(gt_frames_data, pred_frames, dist_thr=0.5)
        results[tracker_name] = {
            "mota": metrics.mota, "idf1": metrics.idf1,
            "id_switches": metrics.id_switches, "fragmentations": metrics.fragmentations,
            "tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn,
        }
        print(f"  [{level_name}/{tracker_name}] MOTA={metrics.mota:.4f} IDF1={metrics.idf1:.4f} "
              f"IDSW={metrics.id_switches} TP={metrics.tp} FP={metrics.fp} FN={metrics.fn}")
    return results


# ── Field construction ──

def build_fields_from_positions(positions_per_frame, velocities_per_frame, sigma_m, bev_down=4, scores_per_frame=None):
    """Build field tensors from per-frame positions and velocities."""
    from temporal.field_builder import build_all_fields
    from temporal.coordinates import grid_shape_reduced

    n_frames = len(positions_per_frame)
    gh, gw = grid_shape_reduced(bev_down)
    all_fields = np.zeros((n_frames, 5, gh, gw), dtype=np.float32)
    for fi in range(n_frames):
        pos = positions_per_frame[fi]
        vel = velocities_per_frame[fi] if velocities_per_frame[fi] is not None else np.zeros_like(pos)
        scores = scores_per_frame[fi] if scores_per_frame is not None else None
        if pos.shape[0] > 0:
            all_fields[fi] = build_all_fields(pos, vel, sigma_m=sigma_m, scores=scores, bev_down=bev_down)
    return all_fields


def compute_velocities_from_positions(positions_per_frame, dt=0.5):
    """Estimate velocities from consecutive frame positions using nearest-neighbor matching."""
    n = len(positions_per_frame)
    velocities = [np.zeros_like(positions_per_frame[i]) for i in range(n)]

    for fi in range(1, n):
        prev_pos = positions_per_frame[fi - 1]
        curr_pos = positions_per_frame[fi]
        if prev_pos.shape[0] == 0 or curr_pos.shape[0] == 0:
            continue

        # Simple nearest-neighbor velocity estimation
        dists = np.linalg.norm(curr_pos[:, None, :] - prev_pos[None, :, :], axis=2)
        vel = np.zeros_like(curr_pos)
        for ci in range(curr_pos.shape[0]):
            pi = np.argmin(dists[ci])
            if dists[ci, pi] < 1.0:  # max 1m displacement at 2Hz = 2m/s
                vel[ci] = (curr_pos[ci] - prev_pos[pi]) / dt
        velocities[fi] = vel

    return velocities


# ── Baseline evaluation ──

def eval_baselines_on_fields(fields, n_history, n_future, sigma_m, label):
    """Evaluate non-learning baselines on prebuilt fields."""
    from temporal.baselines import predict_persistence, predict_field_advection
    from temporal.field_metrics import compute_occupancy_auprc

    n_frames = fields.shape[0]
    n_windows = n_frames - n_history - n_future + 1
    results = {}

    for baseline_name in ["persistence", "advection"]:
        auprc_list = []
        for wi in range(max(n_windows, 1)):
            t_now = n_history + wi - 1
            if t_now + n_future >= n_frames:
                break

            current_occ = fields[t_now, 0]
            current_vx = fields[t_now, 1]
            current_vy = fields[t_now, 2]

            if baseline_name == "persistence":
                preds = predict_persistence(current_occ, n_future)
            elif baseline_name == "advection":
                preds = predict_field_advection(current_occ, current_vx, current_vy, n_future)

            for s in range(n_future):
                gt_occ = fields[t_now + 1 + s, 0]
                auprc = compute_occupancy_auprc(preds[s], gt_occ)
                auprc_list.append(auprc)

        results[baseline_name] = {
            "occ_auprc_mean": float(np.mean(auprc_list)) if auprc_list else 0.0,
            "occ_auprc_std": float(np.std(auprc_list)) if auprc_list else 0.0,
            "n_evaluations": len(auprc_list),
        }
        print(f"  [{label}/{baseline_name}] AUPRC={results[baseline_name]['occ_auprc_mean']:.4f} "
              f"±{results[baseline_name]['occ_auprc_std']:.4f}")

    return results


# ── ConvLSTM training ──

def train_convlstm(annotations_dir, output_dir, device_str, seed, sigma_m, bev_down=4):
    """Train ConvLSTM on GT fields and evaluate."""
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
                                    history_len=4, future_len=4, sigma_m=sigma_m,
                                    bev_down=bev_down)
    val_ds = FieldSequenceDataset(annotations_dir, split="val",
                                  history_len=4, future_len=4, sigma_m=sigma_m,
                                  bev_down=bev_down)
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

    criterion = CombinedTemporalLoss(lambda_vel=0.5, lambda_trace=0.1, ablation="full", bev_down=bev_down)
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


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Run Module 2 three-level evaluation pipeline")
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--detections_jsonl", type=str, default=None,
                        help="Detector JSONL output (enables Level 2 & 3 evaluation)")
    parser.add_argument("--output_dir", type=str, default="outputs/m2_pipeline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sigma_m", type=float, default=0.2)
    parser.add_argument("--bev_down", type=int, default=4,
                        help="BEV grid downsampling (4=0.1m, 8=0.2m, 16=0.4m)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_training", action="store_true")
    args = parser.parse_args()

    ann_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from temporal.annotation_reader import load_all_annotations, build_trajectories, compute_velocities
    from temporal.time_utils import get_split_range, DT

    all_results = {}

    # ────────────────────────────────────────────
    # LEVEL 1: GT positions + GT identity
    # ────────────────────────────────────────────
    banner("LEVEL 1: GT positions + GT identity")

    val_start, val_end = get_split_range("val")
    n_val = val_end - val_start
    frames_val = load_all_annotations(ann_dir, frame_start=val_start, max_frames=n_val)
    trajectories_val = build_trajectories(frames_val)

    # GT positions and IDs
    gt_frames_data = []
    gt_positions_val = []
    for frame_dets in frames_val:
        pos = np.array([[d.world_x_m, d.world_y_m] for d in frame_dets]) if frame_dets else np.empty((0, 2))
        ids = np.array([d.person_id for d in frame_dets], dtype=np.int64) if frame_dets else np.array([], dtype=np.int64)
        gt_frames_data.append({"positions": pos, "ids": ids})
        gt_positions_val.append(pos)

    # L1 Tracking (GT → tracker → evaluate)
    print("\n  [L1] Tracking evaluation...")
    all_results["L1_tracking"] = eval_tracking(gt_frames_data, gt_positions_val, val_start, "L1")

    # L1 Field construction + baselines
    print("\n  [L1] Building GT fields...")
    person_vel = {}
    for pid, traj in trajectories_val.items():
        vel = compute_velocities(traj, dt=DT)
        for i, det in enumerate(traj.detections):
            if det.frame_index not in person_vel:
                person_vel[det.frame_index] = {}
            person_vel[det.frame_index][pid] = vel[i]

    gt_vel_val = []
    for fi, frame_dets in enumerate(frames_val):
        if not frame_dets:
            gt_vel_val.append(np.empty((0, 2)))
            continue
        vels = np.zeros((len(frame_dets), 2), dtype=np.float64)
        for j, d in enumerate(frame_dets):
            vm = person_vel.get(d.frame_index, {})
            if d.person_id in vm:
                vels[j] = vm[d.person_id]
        gt_vel_val.append(vels)

    gt_fields = build_fields_from_positions(gt_positions_val, gt_vel_val, args.sigma_m, args.bev_down)

    print(f"  [L1] Fields: occ_mean={gt_fields[:, 0].mean():.6f}, "
          f"occ_max={gt_fields[:, 0].max():.4f}, "
          f"vel_mean={np.sqrt(gt_fields[:, 1]**2 + gt_fields[:, 2]**2).mean():.4f}")

    print("\n  [L1] Baseline evaluation...")
    all_results["L1_baselines"] = eval_baselines_on_fields(gt_fields, 4, 4, args.sigma_m, "L1")

    # L1 ConvLSTM training
    if not args.skip_training:
        print("\n  [L1] ConvLSTM training...")
        all_results["L1_convlstm"] = train_convlstm(ann_dir, output_dir, args.device, args.seed, args.sigma_m, args.bev_down)
    else:
        all_results["L1_convlstm"] = {"skipped": True}

    # ────────────────────────────────────────────
    # LEVEL 2 & 3: Detector-driven (if JSONL provided)
    # ────────────────────────────────────────────
    if args.detections_jsonl:
        from temporal.detection_loader import (
            load_detections_jsonl, detections_to_positions,
            detections_to_scores, match_detections_to_gt,
        )

        banner("LEVEL 2: Detector positions + GT association")
        det_by_frame = load_detections_jsonl(Path(args.detections_jsonl))
        det_positions_val = detections_to_positions(det_by_frame, val_start, n_val)
        det_scores_val = detections_to_scores(det_by_frame, val_start, n_val)

        # L2: Match detections to GT (Hungarian), assign GT identity
        l2_positions_val = []
        l2_gt_frames = []
        for fi in range(n_val):
            matched_pos, matched_ids = match_detections_to_gt(
                det_positions_val[fi], gt_frames_data[fi]["positions"],
                gt_frames_data[fi]["ids"], dist_thr=0.5
            )
            l2_positions_val.append(matched_pos)
            l2_gt_frames.append({"positions": matched_pos, "ids": matched_ids})

        print("\n  [L2] Tracking evaluation (det positions + GT IDs)...")
        all_results["L2_tracking"] = eval_tracking(gt_frames_data, [f["positions"] for f in l2_gt_frames], val_start, "L2")

        print("\n  [L2] Building detector fields (GT-associated)...")
        l2_vel = compute_velocities_from_positions(l2_positions_val, dt=DT)
        l2_fields = build_fields_from_positions(l2_positions_val, l2_vel, args.sigma_m, args.bev_down)
        print(f"  [L2] Fields: occ_mean={l2_fields[:, 0].mean():.6f}, "
              f"occ_max={l2_fields[:, 0].max():.4f}")

        print("\n  [L2] Baseline evaluation...")
        all_results["L2_baselines"] = eval_baselines_on_fields(l2_fields, 4, 4, args.sigma_m, "L2")

        # ──────────────────────────
        banner("LEVEL 3: Detector positions + Tracker")

        print("\n  [L3] Tracking evaluation (det positions + tracker IDs)...")
        all_results["L3_tracking"] = eval_tracking(gt_frames_data, det_positions_val, val_start, "L3")

        # L3 Diagnostics: identify IDSW and FP root causes
        from temporal.tracker_diagnostics import diagnose_tracker
        from temporal.tracker_kalman import KalmanHungarianTracker

        l3_tracker = KalmanHungarianTracker(dist_gate=1.0, max_age=2, min_hits=2)
        l3_pred_frames = _run_tracker_on_frames(l3_tracker, det_positions_val, val_start)
        diag = diagnose_tracker(gt_frames_data, l3_pred_frames, dist_thr=0.5, frame_offset=val_start)
        print(f"\n  [L3/diag] {diag.summary}")
        all_results["L3_diagnostics"] = {
            "total_idsw": diag.total_idsw,
            "total_fp": diag.total_fp,
            "idsw_by_gt_id": diag.idsw_by_gt_id,
            "fp_by_track_id": {str(k): v for k, v in list(diag.fp_by_track_id.items())[:10]},
            "idsw_events": [
                {"frame": e.frame_index, "gt_id": e.gt_id,
                 "old_track": e.old_track_id, "new_track": e.new_track_id,
                 "dist_to_new": round(e.distance_to_new, 4)}
                for e in diag.id_switch_events
            ],
        }

        # L3 Tracker parameter grid search
        print("\n  [L3] Tracker parameter grid search...")
        from temporal.tracking_metrics import evaluate_tracking as _eval_track
        grid_results = []
        for min_h in [2, 3, 4]:
            for max_a in [1, 2, 3]:
                for dg in [0.5, 0.75, 1.0]:
                    trk = KalmanHungarianTracker(dist_gate=dg, max_age=max_a, min_hits=min_h)
                    pf = _run_tracker_on_frames(trk, det_positions_val, val_start)
                    m = _eval_track(gt_frames_data, pf, dist_thr=0.5)
                    grid_results.append({
                        "min_hits": min_h, "max_age": max_a, "dist_gate": dg,
                        "mota": m.mota, "idf1": m.idf1,
                        "idsw": m.id_switches, "fp": m.fp, "fn": m.fn,
                    })

        grid_results.sort(key=lambda r: -r["mota"])
        all_results["L3_tracker_grid"] = grid_results[:10]
        print(f"  [L3/grid] Top 5 configurations by MOTA:")
        for i, r in enumerate(grid_results[:5]):
            print(f"    #{i+1}: min_hits={r['min_hits']} max_age={r['max_age']} "
                  f"dist_gate={r['dist_gate']} → MOTA={r['mota']:.4f} "
                  f"IDF1={r['idf1']:.4f} IDSW={r['idsw']} FP={r['fp']} FN={r['fn']}")

        print("\n  [L3] Building detector+tracker fields...")
        l3_vel = compute_velocities_from_positions(det_positions_val, dt=DT)
        l3_fields = build_fields_from_positions(det_positions_val, l3_vel, args.sigma_m, args.bev_down,
                                                scores_per_frame=det_scores_val)
        print(f"  [L3] Fields: occ_mean={l3_fields[:, 0].mean():.6f}, "
              f"occ_max={l3_fields[:, 0].max():.4f}")

        print("\n  [L3] Baseline evaluation...")
        all_results["L3_baselines"] = eval_baselines_on_fields(l3_fields, 4, 4, args.sigma_m, "L3")
    else:
        print("\n[INFO] No --detections_jsonl provided. Skipping Level 2 & 3.")
        print("[INFO] To enable: run export_detections_jsonl.py first, then pass the JSONL path.")

    # ── Trajectory Prediction Baseline ──
    banner("TRAJECTORY PREDICTION (constant velocity)")
    from temporal.trajectory_predictor import evaluate_trajectory_baseline

    traj_results = evaluate_trajectory_baseline(
        trajectories_val, split="val", n_history=4, n_future=4, dt=DT
    )
    all_results["trajectory_baseline"] = traj_results
    print(f"  ADE = {traj_results['ade_mean']:.4f} ± {traj_results['ade_std']:.4f} m")
    print(f"  FDE = {traj_results['fde_mean']:.4f} ± {traj_results['fde_std']:.4f} m")
    print(f"  Horizon = {traj_results['horizon_s']:.2f} s")
    print(f"  N_trajectories = {traj_results['n_trajectories']}, N_windows = {traj_results['n_windows']}")

    # ── MLP Trajectory Predictor ──
    banner("TRAJECTORY PREDICTION (MLP)")
    from temporal.mlp_predictor import (
        extract_trajectory_windows, train_mlp_predictor,
        evaluate_mlp_predictor, MLPTrajectoryConfig,
    )
    from temporal.annotation_reader import load_all_annotations, build_trajectories as _build_traj

    train_start, train_end = get_split_range("train")
    train_frames = load_all_annotations(ann_dir, frame_start=train_start, max_frames=train_end - train_start)
    all_trajectories = _build_traj(train_frames + frames_val)

    train_hist, train_fut = extract_trajectory_windows(all_trajectories, "train", 4, 4)
    val_hist, val_fut = extract_trajectory_windows(all_trajectories, "val", 4, 4)
    print(f"  Train windows: {train_hist.shape[0]}, Val windows: {val_hist.shape[0]}")

    if train_hist.shape[0] > 0 and val_hist.shape[0] > 0:
        mlp_results_all = []
        for seed in range(5):
            cfg = MLPTrajectoryConfig(
                n_history=4, n_future=4, hidden_dim=64,
                learning_rate=1e-3, weight_decay=1e-4,
                max_epochs=300, patience=30, seed=seed,
            )
            res = train_mlp_predictor(train_hist, train_fut, val_hist, val_fut, cfg)
            mlp_results_all.append(res)
            print(f"  [seed={seed}] val ADE={res['best_val_ade']:.4f}m "
                  f"(epochs={res['epochs_trained']})")

        ades = [r["best_val_ade"] for r in mlp_results_all]
        all_results["trajectory_mlp"] = {
            "val_ade_mean": float(np.mean(ades)),
            "val_ade_std": float(np.std(ades)),
            "val_ade_seeds": ades,
            "n_train": int(train_hist.shape[0]),
            "n_val": int(val_hist.shape[0]),
            "beats_baseline": float(np.mean(ades)) < traj_results["ade_mean"],
        }
        print(f"\n  MLP mean ADE = {np.mean(ades):.4f} ± {np.std(ades):.4f} m")
        print(f"  Const-vel ADE = {traj_results['ade_mean']:.4f} m")
        print(f"  MLP {'BEATS' if np.mean(ades) < traj_results['ade_mean'] else 'DOES NOT BEAT'} baseline")
    else:
        print("  [SKIP] Insufficient trajectory windows for MLP training")
        all_results["trajectory_mlp"] = {"skipped": True}

    # ── Summary ──
    banner("RESULTS SUMMARY")
    results_path = output_dir / "m2_pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print("\n  Tracking comparison (val split, Kalman tracker):")
    for level in ["L1", "L2", "L3"]:
        key = f"{level}_tracking"
        if key in all_results and "kalman" in all_results[key]:
            m = all_results[key]["kalman"]
            print(f"    {level}: MOTA={m['mota']:.4f} IDF1={m['idf1']:.4f} "
                  f"IDSW={m['id_switches']} TP={m['tp']} FP={m['fp']} FN={m['fn']}")

    print("\n  Baselines comparison (val AUPRC, persistence):")
    for level in ["L1", "L2", "L3"]:
        key = f"{level}_baselines"
        if key in all_results and "persistence" in all_results[key]:
            m = all_results[key]["persistence"]
            print(f"    {level}: {m['occ_auprc_mean']:.4f} ±{m['occ_auprc_std']:.4f}")

    if "L1_convlstm" in all_results and "best_auprc" in all_results.get("L1_convlstm", {}):
        print(f"\n  ConvLSTM (L1, best val AUPRC): {all_results['L1_convlstm']['best_auprc']:.4f}")

    if "trajectory_baseline" in all_results:
        tb = all_results["trajectory_baseline"]
        print(f"\n  Trajectory baseline (constant velocity, {tb['horizon_s']:.1f}s):")
        print(f"    ADE={tb['ade_mean']:.4f}m  FDE={tb['fde_mean']:.4f}m  ({tb['n_trajectories']} predictions)")

    if "trajectory_mlp" in all_results and "val_ade_mean" in all_results.get("trajectory_mlp", {}):
        tm = all_results["trajectory_mlp"]
        print(f"  Trajectory MLP (5 seeds):")
        print(f"    ADE={tm['val_ade_mean']:.4f} ± {tm['val_ade_std']:.4f}m  "
              f"({'BEATS' if tm['beats_baseline'] else 'DOES NOT BEAT'} baseline)")

    print("\n[DONE] Pipeline complete", flush=True)


if __name__ == "__main__":
    main()
