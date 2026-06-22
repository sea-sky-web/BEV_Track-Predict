#!/usr/bin/env python3
"""
Colab 训练启动脚本。

前置条件（由 GitHub Actions workflow 或手动完成）：
    colab new --gpu T4 -s bev-train
    colab drivemount -s bev-train

用法：
    colab exec -s bev-train -f scripts/colab_train.py --timeout 18000 \
        -- --data_zip /content/drive/MyDrive/Colab_Notebooks/dataSet/wildtrack.zip
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/sea-sky-web/BEV_Track-Predict.git"
REPO_DIR = Path("/content/BEV_Track-Predict")


def run(cmd, cwd=None, check=True):
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check).returncode


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


parser = argparse.ArgumentParser()
parser.add_argument("--data_zip",
                    default="/content/drive/MyDrive/Colab_Notebooks/dataSet/wildtrack.zip",
                    help="Drive 上 wildtrack.zip 的绝对路径")
parser.add_argument("--data_root", default=None,
                    help="已解压的数据集路径（优先于 --data_zip）")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

# ── 1. 克隆 / 更新仓库 ────────────────────────────────────────
banner("1/5  克隆 / 更新仓库")
if REPO_DIR.exists():
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

os.chdir(str(REPO_DIR))

# ── 2. 定位 / 解压数据集 ──────────────────────────────────────
banner("2/5  数据集准备")
data_root = REPO_DIR / "wildtrack"

if args.data_root:
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] 指定的 data_root 不存在：{data_root}")
        sys.exit(1)
    print(f"[OK] 使用已有数据：{data_root}")
elif data_root.is_dir() and (data_root / "Image_subsets").exists():
    print(f"[OK] 数据已存在：{data_root}")
else:
    # 清理可能存在的错误文件/目录
    run(["rm", "-rf", str(REPO_DIR / "wildtrack"), str(REPO_DIR / "wiltrack")], check=False)

    zip_path = Path(args.data_zip)
    if not zip_path.exists():
        print(f"[ERROR] zip 不存在：{zip_path}")
        print("请确认 Google Drive 已挂载且路径正确")
        sys.exit(1)

    print(f"[OK] 解压 {zip_path} ...")
    run(["unzip", "-q", str(zip_path), "-d", str(REPO_DIR)])
    print(f"[OK] 解压完成")

# ── 3. 验证数据集 ─────────────────────────────────────────────
banner("3/5  验证数据集结构")
for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
    p = data_root / subdir
    if p.exists():
        print(f"  [OK] {subdir}")
    else:
        print(f"  [ERROR] 缺少：{subdir}")
        # 看看解压出了什么
        run(["ls", "-la", str(REPO_DIR)], check=False)
        sys.exit(1)

# ── 4. 安装依赖 ───────────────────────────────────────────────
banner("4/5  安装 Python 依赖")
run([sys.executable, "-m", "pip", "install", "-q",
     "-r", str(REPO_DIR / "requirements.txt")])

# ── 5. 训练 ──────────────────────────────────────────────────
banner("5/5  开始训练")

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
    "--device",                 args.device,
    "--log_every",              "20",
]

print("命令：", " ".join(train_cmd))
ret = run(train_cmd, cwd=str(REPO_DIR), check=False)
if ret != 0:
    sys.exit(ret)

print(f"\n[OK] 训练完成")
print(f"模型：{REPO_DIR / 'outputs/train_multicam_mvdet_style_v3/model_final.pth'}")
