#!/usr/bin/env python3
"""
Colab 评估启动脚本。

在 Colab 上运行 MODA/MODP 检测评估 + 可视化。
通过 GitHub Actions colab-eval.yml 调用。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/sea-sky-web/BEV_Track-Predict.git"
REPO_DIR = Path("/content/BEV_Track-Predict")
GDRIVE_FOLDER_ID = "1uBptJBbtMzVRQwSMRbQkIJp8-VVoBqUK"


def run(cmd, cwd=None, check=True):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"\n>>> {' '.join(str(c) for c in cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}", flush=True)


def download_dataset(data_root: Path, gdrive_id: str) -> Path:
    """Download WildTrack dataset using the same logic as colab_train.py."""
    if data_root.is_dir() and (data_root / "Image_subsets").exists():
        print(f"[OK] 数据已存在：{data_root}")
        return data_root

    run(["rm", "-rf", str(REPO_DIR / "wildtrack"), str(REPO_DIR / "wiltrack")], check=False)

    print(f"[INFO] 从 Google Drive 文件夹下载 wildtrack (ID: {gdrive_id}) ...")
    run([sys.executable, "-m", "pip", "install", "-q", "gdown==5.2.0"])
    ret = run([
        sys.executable, "-m", "gdown", "--folder", "--fuzzy", "--remaining-ok",
        f"https://drive.google.com/drive/folders/{gdrive_id}",
        "-O", str(REPO_DIR),
    ], check=False)

    if ret != 0:
        print("[WARN] gdown --folder 下载失败，尝试 fallback 路径 ...")
        fallback_paths = [
            Path("/content/wildtrack"),
            Path("/content/BEV_Track-Predict/wildtrack"),
            Path("/root/wildtrack"),
        ]
        import shutil
        for fb in fallback_paths:
            if fb.is_dir() and (fb / "Image_subsets").exists():
                if data_root.exists():
                    shutil.rmtree(str(data_root))
                shutil.copytree(str(fb), str(data_root))
                print(f"[OK] 使用 fallback 数据：{fb}")
                return data_root
        print("[ERROR] gdown 下载失败且无可用 fallback 数据")
        sys.exit(1)

    zip_path = REPO_DIR / "wildtrack.zip"
    if zip_path.exists():
        print(f"[INFO] 发现 wildtrack.zip ({zip_path.stat().st_size / 1e9:.2f} GB)，解压中 ...")
        run(["unzip", "-q", str(zip_path), "-d", str(REPO_DIR)])
        zip_path.unlink()
        print("[OK] 解压完成")

    found = list(REPO_DIR.rglob("Image_subsets"))
    if not found:
        print("[ERROR] 找不到 Image_subsets，下载或解压失败")
        sys.exit(1)
    actual_root = found[0].parent
    if actual_root != data_root:
        import shutil
        if data_root.exists():
            shutil.rmtree(str(data_root))
        actual_root.rename(data_root)
        print(f"[OK] 数据移动到：{data_root}")

    print(f"[OK] 数据就绪：{data_root}")
    return data_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/train_multicam_mvdet_style_v3/model_final.pth")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gdrive_id", default=GDRIVE_FOLDER_ID)
    parser.add_argument("--fusion_mode", default="geo_confidence_v1")
    parser.add_argument("--backbone", default="mobilenet_v2")
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--frame_start", type=int, default=360)
    parser.add_argument("--max_frames", type=int, default=40)
    args = parser.parse_args()

    # 1. Clone / pull
    banner("1/5  克隆 / 更新仓库")
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        if REPO_DIR.exists():
            import shutil
            saved = {}
            outputs = REPO_DIR / "outputs"
            if outputs.exists():
                tmp_out = Path("/tmp/_bev_outputs_save")
                if tmp_out.exists():
                    shutil.rmtree(tmp_out)
                shutil.move(str(outputs), str(tmp_out))
                saved["outputs"] = tmp_out
            shutil.rmtree(str(REPO_DIR))
            run(["git", "clone", REPO_URL, str(REPO_DIR)])
            for name, tmp in saved.items():
                dest = REPO_DIR / name
                if not dest.exists():
                    shutil.move(str(tmp), str(dest))
        else:
            run(["git", "clone", REPO_URL, str(REPO_DIR)])

    # 2. Download dataset
    banner("2/5  数据集准备")
    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.exists():
            print(f"[ERROR] 指定的 data_root 不存在：{data_root}")
            sys.exit(1)
    else:
        data_root = download_dataset(REPO_DIR / "wildtrack", args.gdrive_id)

    # 3. Verify dataset
    banner("3/5  验证数据集结构")
    for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
        p = data_root / subdir
        if p.exists():
            print(f"  [OK] {subdir}")
        else:
            print(f"  [ERROR] 缺少：{subdir}")
            sys.exit(1)

    # 4. Install deps
    banner("4/5  安装 Python 依赖")
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)

    # 5. Evaluate
    banner("5/5  检测评估 + 可视化")
    model_path = REPO_DIR / args.model_path
    eval_out = REPO_DIR / "outputs" / "eval_results.json"

    eval_cmd = [
        sys.executable, str(REPO_DIR / "src" / "evaluate_main.py"),
        "--data_root", str(data_root),
        "--model_path", str(model_path),
        "--device", args.device,
        "--report_detection",
        "--metrics_out", str(eval_out),
        "--views", "0,1,2,3,4,5,6",
        "--fusion_mode", args.fusion_mode,
        "--backbone", args.backbone,
        "--frame_start", str(args.frame_start),
        "--max_frames", str(args.max_frames),
        "--det_thresholds=-0.50,-0.25,-0.10,0.00,0.05,0.10,0.15,0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50,0.55,0.60",
        "--det_min_distances=3.0,4.0,5.0,6.0,7.0,8.0",
        "--loss_type", args.loss_type,
    ]

    print("\n[EVAL] 开始评估...", flush=True)
    print("命令：", " ".join(eval_cmd), flush=True)
    eval_ret = run(eval_cmd, cwd=str(REPO_DIR), check=False)
    print(f"\n[EVAL] exit code: {eval_ret}", flush=True)

    if eval_out.exists():
        r = json.loads(eval_out.read_text())
        print("\n=== EVAL RESULTS (inline) ===")
        for k in ["det_moda", "det_modp", "det_precision", "det_recall", "det_f1",
                   "det_best_threshold", "det_best_nms_radius",
                   "det_moda_tp", "det_moda_fp", "det_moda_fn"]:
            print(f"{k}: {r.get(k, 'N/A')}")
        print("=== END EVAL RESULTS ===", flush=True)
    else:
        print("[WARN] eval_results.json not found", flush=True)

    # Visualization
    viz_cmd = [
        sys.executable, str(REPO_DIR / "scripts" / "visualize_prediction.py"),
        "--model_path", str(model_path),
        "--data_root", str(data_root),
        "--output", str(REPO_DIR / "outputs" / "visualization"),
        "--device", args.device,
        "--frame", "0",
        "--backbone", args.backbone,
        "--fusion_mode", args.fusion_mode,
    ]
    print("\n[VIZ] 生成可视化...", flush=True)
    run(viz_cmd, cwd=str(REPO_DIR), check=False)

    print("\n[OK] 全部完成", flush=True)


if __name__ == "__main__":
    main()
