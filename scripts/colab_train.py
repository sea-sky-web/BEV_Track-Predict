#!/usr/bin/env python3
"""
Colab 训练启动脚本。

用法：
    # 本地 CLI（手动）
    colab run --gpu T4 --keep --session bev-train --timeout 18000 \
        scripts/colab_train.py --epochs 10

    # GitHub Actions 通过 workflow_dispatch 触发
    colab run --gpu T4 --keep --session bev-train --timeout 18000 \
        scripts/colab_train.py \
        --data_root /content/drive/MyDrive/wildtrack \
        --epochs 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/sea-sky-web/BEV_Track-Predict.git"
REPO_DIR = Path("/content/BEV_Track-Predict")

DRIVE_CANDIDATES = [
    "/content/drive/MyDrive/wildtrack",
    "/content/drive/MyDrive/datasets/wildtrack",
    "/content/drive/MyDrive/Wildtrack",
    "/content/drive/MyDrive/WildTrack",
]


def run(cmd, cwd=None, check=True):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check).returncode


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


# ── CLI 参数 ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default=None,
                    help="数据集绝对路径。不填时从 Drive 候选路径自动检测。")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--gpu", default="cuda",
                    help="'cuda' 或 'cpu'，传给 train_main.py --device")
args = parser.parse_args()

# ── 1. 克隆 / 更新仓库 ────────────────────────────────────────
banner("1/5  克隆 / 更新仓库")
if REPO_DIR.exists():
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

# ── 2. 挂载 Drive（最佳努力）并确定数据集路径 ─────────────────
banner("2/5  数据集定位")

if args.data_root:
    # 如果 data_root 在 Drive 下，先尝试挂载
    if str(args.data_root).startswith("/content/drive"):
        try:
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
            print("[OK] Google Drive 已挂载")
        except Exception as e:
            print(f"[WARN] Drive 挂载失败：{e}")
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] 指定的 data_root 不存在：{data_root}")
        sys.exit(1)
    print(f"[OK] 使用指定路径：{data_root}")
else:
    # 自动检测：先挂载 Drive
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        print("[OK] Google Drive 已挂载")
    except Exception as e:
        print(f"[WARN] Drive 挂载失败：{e}")

    dataset_src = None
    for candidate in DRIVE_CANDIDATES:
        if Path(candidate).exists():
            dataset_src = candidate
            print(f"[OK] 找到数据集：{candidate}")
            break

    if dataset_src is None:
        print("[ERROR] 未找到 wildtrack 数据集，候选路径：")
        for c in DRIVE_CANDIDATES:
            print(f"  {c}")
        sys.exit(1)

    data_root = REPO_DIR / "wildtrack"
    if not data_root.exists():
        run(["ln", "-s", dataset_src, str(data_root)])
        print(f"[OK] 软链接：{data_root} -> {dataset_src}")

# ── 3. 安装依赖 ───────────────────────────────────────────────
banner("3/5  安装 Python 依赖")
run([sys.executable, "-m", "pip", "install", "-q",
     "-r", str(REPO_DIR / "requirements.txt")])
print("[OK] 依赖安装完成")

# ── 4. 验证数据集 ─────────────────────────────────────────────
banner("4/5  验证数据集结构")
for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
    p = data_root / subdir
    if p.exists():
        print(f"  [OK] {subdir}")
    else:
        print(f"  [ERROR] 缺少：{subdir}")
        sys.exit(1)

# ── 5. 训练 ──────────────────────────────────────────────────
banner("5/5  开始训练")
os.chdir(str(REPO_DIR))

train_cmd = [
    sys.executable, "scripts/train_main.py",
    "--data_root",              str(data_root),
    "--views",                  "0,1,2,3,4,5,6",
    "--max_frames",             "-1",
    "--epochs",                 str(args.epochs),
    "--batch",                  "1",
    "--pretrained",             "true",
    "--backbone",               "resnet18",
    "--fusion_mode",            "confidence_v2",
    "--augment",                "true",
    "--augment_hflip_prob",     "0.0",
    "--augment_color_jitter",   "0.2,0.2,0.2,0.05",
    "--alpha",                  "1.0",
    "--optimizer",              "adam",
    "--scheduler",              "cosine",
    "--lr_init",                "0.0001",
    "--weight_decay",           "0.0001",
    "--freeze_backbone_epochs", "3",
    "--device",                 args.gpu,
    "--log_every",              "20",
]

print("命令：", " ".join(train_cmd))
ret = run(train_cmd, cwd=str(REPO_DIR), check=False)
if ret != 0:
    sys.exit(ret)

out = REPO_DIR / "outputs" / "train_multicam_mvdet_style_v3" / "model_final.pth"
print(f"\n[OK] 训练完成，模型：{out}")
