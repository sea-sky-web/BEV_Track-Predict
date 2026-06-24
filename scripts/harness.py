#!/usr/bin/env python3
"""
harness.py — Research-iteration CLI for BEV pedestrian detection.

Commands:
  sanity   Quick local check before burning Colab GPU
  train    Trigger GitHub Actions Colab training workflow
  watch    Poll a run until done, download artifacts, auto-inspect
  inspect  Read eval_results.json + metrics.csv → ALIGNED/IMPROVING/STUCK/FAILED
  history  Cross-run MODA/SNR trend table from ai_runs/
  loop     Automated train→watch→inspect cycle until MODA target
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKFLOW = "colab-train.yml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_RUNS_DIR = PROJECT_ROOT / "ai_runs"


# ── gh CLI wrapper ────────────────────────────────────────────

def _gh(*args, capture=True):
    cmd = ["gh"] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args)} failed:\n{r.stderr.strip()}")
        return r.stdout.strip()
    subprocess.check_call(cmd)
    return ""


def _latest_run_id():
    out = _gh("run", "list", "--workflow", WORKFLOW, "--limit", "1", "--json", "databaseId")
    data = json.loads(out)
    if not data:
        raise RuntimeError("No runs found for " + WORKFLOW)
    return int(data[0]["databaseId"])


def _run_info(run_id):
    out = _gh("run", "view", str(run_id), "--json",
              "databaseId,status,conclusion,url,updatedAt,number")
    return json.loads(out)


# ── sanity ────────────────────────────────────────────────────

def cmd_sanity(args):
    """Quick local check: imports, forward pass, optional 2-frame mini-train."""
    print("[sanity] checking imports …")
    try:
        subprocess.check_call(
            [sys.executable, "-c",
             "import torch; from src.models import build_model; "
             "m = build_model(); "
             "x = torch.randn(1, 7, 3, 720, 1280); "
             "out = m(x); print(f'output shape: {out.shape}')"],
            cwd=str(PROJECT_ROOT))
        print("[sanity] forward pass OK")
    except subprocess.CalledProcessError:
        print("[sanity] FAILED — fix import / model errors before training")
        sys.exit(1)

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "wildtrack"
    if (data_root / "Image_subsets").exists():
        print(f"[sanity] data found at {data_root}, running 2-frame 1-epoch micro-train …")
        ret = subprocess.call(
            [sys.executable, "scripts/train_main.py",
             "--data_root", str(data_root),
             "--views", "0,1,2,3,4,5,6",
             "--max_frames", "2",
             "--epochs", "1",
             "--batch", "1",
             "--backbone", "resnet18",
             "--fusion_mode", "confidence_v2",
             "--bev_pos_weight", "10.0",
             "--device", "cpu",
             "--log_every", "1"],
            cwd=str(PROJECT_ROOT))
        if ret == 0:
            print("[sanity] micro-train OK")
        else:
            print("[sanity] micro-train FAILED")
            sys.exit(1)
    else:
        print(f"[sanity] no data at {data_root} — skipping micro-train (import-only check passed)")

    print("[sanity] ALL OK")


# ── train ─────────────────────────────────────────────────────

def cmd_train(args):
    """Trigger GitHub Actions workflow dispatch."""
    print(f"[train] epochs={args.epochs} max_frames={args.max_frames} "
          f"pos_weight={args.pos_weight} gpu={args.gpu}")
    _gh("workflow", "run", WORKFLOW,
        "-f", f"epochs={args.epochs}",
        "-f", f"bev_pos_weight={args.pos_weight}",
        "-f", f"gpu={args.gpu}",
        "-f", f"max_frames={args.max_frames}")
    time.sleep(8)
    run_id = _latest_run_id()
    info = _run_info(run_id)
    print(f"[train] triggered run {run_id} (#{info.get('number')})")
    print(f"[train] {info.get('url')}")
    return run_id


# ── watch ─────────────────────────────────────────────────────

def cmd_watch(args):
    """Poll until run completes, download artifacts, run inspect."""
    run_id = args.run_id or _latest_run_id()
    poll = args.poll
    out_dir = Path(args.out)
    print(f"[watch] monitoring run {run_id} (poll={poll}s) …")

    while True:
        info = _run_info(run_id)
        st = info.get("status", "?")
        co = info.get("conclusion") or "-"
        print(f"  [{st}] conclusion={co}  updated={str(info.get('updatedAt', ''))[:19]}")
        if st == "completed":
            break
        time.sleep(poll)

    print(f"[watch] run finished: conclusion={info.get('conclusion')}")

    # Download artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    run_num = info.get("number")
    artifact_name = f"bev-checkpoint-run{run_num}"
    try:
        print(f"[watch] downloading {artifact_name} → {out_dir}/")
        _gh("run", "download", str(run_id), "-n", artifact_name, "-D", str(out_dir))
    except Exception as e:
        print(f"[watch] download warning: {e}")

    # Auto-inspect
    print()
    _do_inspect(out_dir)


# ── inspect ───────────────────────────────────────────────────

def _read_metrics_csv(path: Path):
    """Read metrics.csv → list of dicts, one per epoch."""
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _do_inspect(out_dir: Path, moda_target: float = 0.3):
    """Core inspect logic → prints verdict and returns status string."""
    eval_path = out_dir / "eval_results.json"
    metrics_path = out_dir / "metrics.csv"

    # Check eval results
    moda = None
    if eval_path.exists():
        r = json.loads(eval_path.read_text())
        print("=" * 52)
        print("  EVALUATION RESULTS")
        print("=" * 52)
        for label, key in [
            ("MODA", "det_moda"), ("MODP", "det_modp"),
            ("Precision", "det_precision"), ("Recall", "det_recall"),
            ("F1", "det_f1"), ("Threshold", "det_best_threshold"),
        ]:
            v = r.get(key)
            if v is not None:
                print(f"  {label:12s}: {v:.4f}")
        tp, fp, fn = r.get("det_moda_tp"), r.get("det_moda_fp"), r.get("det_moda_fn")
        if tp is not None:
            print(f"  TP/FP/FN    : {int(tp)}/{int(fp)}/{int(fn)}")
        print("=" * 52)
        moda = r.get("det_moda")

    # Check training metrics
    rows = _read_metrics_csv(metrics_path)
    snr_last = None
    if rows:
        print(f"\n  Training: {len(rows)} epochs")
        first_loss = float(rows[0].get("loss", 0))
        last_loss = float(rows[-1].get("loss", 0))
        snr_last = float(rows[-1].get("snr", 0)) if "snr" in rows[-1] else None
        print(f"  Loss: {first_loss:.4f} → {last_loss:.4f} (Δ={last_loss - first_loss:.4f})")
        if snr_last is not None:
            print(f"  SNR (last epoch): {snr_last:.3f}")
        # Check saturation (last 2 epochs)
        if len(rows) >= 2:
            prev_loss = float(rows[-2].get("loss", 0))
            delta = abs(last_loss - prev_loss)
            if delta < 0.002:
                print(f"  ⚠ Loss plateau: Δ={delta:.4f} (last 2 epochs)")

    # Verdict
    print()
    if not eval_path.exists():
        verdict = "FAILED"
        print(f"  ▸ {verdict}: eval_results.json missing — eval step did not produce output")
        print("    → Check workflow logs, ensure eval runs in same Colab session")
    elif moda is not None and moda >= moda_target:
        verdict = "ALIGNED"
        print(f"  ▸ {verdict}: MODA={moda:.4f} ≥ target {moda_target}")
        print("    → Check bev_prediction.png for visual confirmation")
    elif snr_last is not None and snr_last > 0:
        verdict = "IMPROVING"
        moda_str = f"{moda:.4f}" if moda is not None else "N/A"
        print(f"  ▸ {verdict}: SNR={snr_last:.3f} > 0, MODA={moda_str} < {moda_target}")
        print("    → Continue training: more epochs or more max_frames")
    else:
        verdict = "STUCK"
        print(f"  ▸ {verdict}: SNR≤0 or missing, model not learning")
        print("    → Try: increase pos_weight, adjust lr, check data pipeline")

    print()
    return verdict


def cmd_inspect(args):
    out_dir = Path(args.dir)
    _do_inspect(out_dir, moda_target=args.moda_target)


# ── history ───────────────────────────────────────────────────

def cmd_history(args):
    """Print cross-run trend table from ai_runs/ context files."""
    runs_dir = Path(args.ai_runs)
    if not runs_dir.exists():
        print(f"[history] {runs_dir} not found")
        return

    entries = []
    for ctx_file in sorted(runs_dir.glob("*/ai_context.md")):
        run_name = ctx_file.parent.name
        text = ctx_file.read_text()

        # Extract last epoch SNR from markdown table
        snr = None
        for line in text.split("\n"):
            if "|" in line and "**" in line:
                parts = [p.strip().strip("*") for p in line.split("|")]
                nums = [p for p in parts if p.replace(".", "").replace("-", "").isdigit()]
                if nums:
                    try:
                        snr = float(nums[-1])
                    except ValueError:
                        pass

        # Extract MODA if mentioned
        moda = None
        for line in text.split("\n"):
            if "MODA" in line and "≥" not in line and "target" not in line.lower():
                for word in line.split():
                    try:
                        v = float(word)
                        if -1 <= v <= 1:
                            moda = v
                            break
                    except ValueError:
                        continue

        entries.append({"run": run_name, "snr": snr, "moda": moda})

    if not entries:
        print("[history] no runs found in ai_runs/")
        return

    print(f"\n{'Run':<30s} {'SNR':>8s} {'MODA':>8s}")
    print("-" * 48)
    for e in entries:
        snr_s = f"{e['snr']:.3f}" if e["snr"] is not None else "—"
        moda_s = f"{e['moda']:.4f}" if e["moda"] is not None else "—"
        print(f"  {e['run']:<28s} {snr_s:>8s} {moda_s:>8s}")
    print()


# ── loop ──────────────────────────────────────────────────────

def cmd_loop(args):
    """Automated train→watch→inspect cycle until MODA target."""
    print(f"[loop] target MODA={args.moda_target}  max_runs={args.max_runs}")
    out_dir = Path(args.out)

    for i in range(1, args.max_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"[loop] iteration {i}/{args.max_runs}")
        print(f"{'=' * 60}")

        # Train
        _gh("workflow", "run", WORKFLOW,
            "-f", f"epochs={args.epochs}",
            "-f", f"bev_pos_weight={args.pos_weight}",
            "-f", f"gpu={args.gpu}",
            "-f", f"max_frames={args.max_frames}")
        time.sleep(8)
        run_id = _latest_run_id()
        info = _run_info(run_id)
        print(f"[loop] triggered run {run_id}  {info.get('url')}")

        # Watch
        while True:
            info = _run_info(run_id)
            if info.get("status") == "completed":
                break
            time.sleep(args.poll)

        # Download
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _gh("run", "download", str(run_id), "-n",
                f"bev-checkpoint-run{info.get('number')}", "-D", str(out_dir))
        except Exception as e:
            print(f"[loop] download warning: {e}")

        # Inspect
        verdict = _do_inspect(out_dir, moda_target=args.moda_target)

        if verdict == "ALIGNED":
            print(f"\n[loop] TARGET MET after {i} run(s). Done.")
            return

        eval_path = out_dir / "eval_results.json"
        moda = None
        if eval_path.exists():
            moda = json.loads(eval_path.read_text()).get("det_moda")

        if moda is not None:
            print(f"[loop] MODA={moda:.4f} < {args.moda_target} — continuing")
        else:
            print("[loop] MODA unavailable — continuing")

    print(f"\n[loop] max_runs={args.max_runs} reached without meeting MODA target")


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="harness",
        description="BEV pedestrian detection — research iteration CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # sanity
    s = sub.add_parser("sanity", help="Quick local check before Colab")
    s.add_argument("--data_root", default=None)

    # train
    t = sub.add_parser("train", help="Trigger Colab training run")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--max_frames", type=int, default=100)
    t.add_argument("--pos_weight", type=float, default=10.0)
    t.add_argument("--gpu", default="T4")

    # watch
    w = sub.add_parser("watch", help="Watch run → download → inspect")
    w.add_argument("--run_id", type=int, default=None)
    w.add_argument("--poll", type=int, default=60)
    w.add_argument("--out", default="artifacts")

    # inspect
    ins = sub.add_parser("inspect", help="Analyze results: ALIGNED/IMPROVING/STUCK/FAILED")
    ins.add_argument("--dir", default="artifacts")
    ins.add_argument("--moda_target", type=float, default=0.3)

    # history
    h = sub.add_parser("history", help="Cross-run trend table from ai_runs/")
    h.add_argument("--ai_runs", default=str(AI_RUNS_DIR))

    # loop
    lp = sub.add_parser("loop", help="Auto train→inspect cycle until MODA target")
    lp.add_argument("--epochs", type=int, default=20)
    lp.add_argument("--max_frames", type=int, default=200)
    lp.add_argument("--pos_weight", type=float, default=10.0)
    lp.add_argument("--gpu", default="T4")
    lp.add_argument("--max_runs", type=int, default=5)
    lp.add_argument("--moda_target", type=float, default=0.3)
    lp.add_argument("--poll", type=int, default=60)
    lp.add_argument("--out", default="artifacts")

    args = ap.parse_args()

    cmds = {
        "sanity": cmd_sanity,
        "train": cmd_train,
        "watch": cmd_watch,
        "inspect": cmd_inspect,
        "history": cmd_history,
        "loop": cmd_loop,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
