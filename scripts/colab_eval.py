#!/usr/bin/env python3
"""
Colab 评估启动脚本。

在 Colab 上运行 MODA/MODP 检测评估 + 可视化。
通过 GitHub Actions colab-eval.yml 调用。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/sea-sky-web/BEV_Track-Predict.git"
REPO_DIR = Path("/content/BEV_Track-Predict")
GDRIVE_FILE_ID = "1LDNFgAEq9wYWkbOPk4UdXQetBkhZSVfy"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/train_multicam_mvdet_style_v3/model_final.pth")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 1. Clone / pull
    if REPO_DIR.exists():
        run("git pull", cwd=str(REPO_DIR))
    else:
        run(f"git clone {REPO_URL} {REPO_DIR}")

    # 2. 下载数据集
    data_root = Path(args.data_root) if args.data_root else REPO_DIR / "wildtrack"
    if not (data_root / "Image_subsets").exists():
        run("pip install -q gdown")
        zip_path = "/tmp/wildtrack.zip"
        run(f"gdown {GDRIVE_FILE_ID} -O {zip_path}")
        print(f"[OK] 下载完成：{zip_path} ({os.path.getsize(zip_path) / 1e9:.1f} GB)", flush=True)
        run(f"unzip -q -o {zip_path} -d {REPO_DIR}")
        os.remove(zip_path)
        print("[OK] 解压完成", flush=True)

    # 3. 安装依赖
    req = REPO_DIR / "requirements.txt"
    if req.exists():
        run(f"pip install -q -r {req}", check=False)
    run("pip install -q torch torchvision numpy opencv-python-headless scipy", check=False)

    # 4. 运行评估
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
    ]
    print("\n[EVAL] 开始评估...", flush=True)
    run(eval_cmd, cwd=str(REPO_DIR / "src"))

    # 5. 生成可视化
    viz_cmd = [
        sys.executable, str(REPO_DIR / "scripts" / "visualize_prediction.py"),
        "--model_path", str(model_path),
        "--data_root", str(data_root),
        "--output", str(REPO_DIR / "outputs" / "visualization"),
        "--device", args.device,
        "--frame", "0",
    ]
    print("\n[VIZ] 生成可视化...", flush=True)
    run(viz_cmd, cwd=str(REPO_DIR), check=False)

    print("\n[OK] 评估完成", flush=True)


if __name__ == "__main__":
    main()
