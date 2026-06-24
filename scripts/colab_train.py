#!/usr/bin/env python3
"""
Colab 训练启动脚本。

用法：
    # GitHub Actions（通过 stdin 注入 sys.argv）
    echo "import sys; sys.argv = [...]" | cat - scripts/colab_train.py | colab exec -s bev-train

    # Colab Notebook cell
    !python /content/BEV_Track-Predict/scripts/colab_train.py
"""
from __future__ import annotations

import argparse
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
    proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def banner(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


parser = argparse.ArgumentParser()
parser.add_argument("--gdrive_id", default=GDRIVE_FOLDER_ID,
                    help="wildtrack 数据集的 Google Drive 文件夹 ID")
parser.add_argument("--data_root", default=None,
                    help="已解压的数据集路径（跳过下载）")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bev_pos_weight", type=float, default=10.0,
                    help="BEV 正样本损失权重")
parser.add_argument("--device", default="cuda")
parser.add_argument("--max_frames", type=int, default=100,
                    help="Max training frames (more data = better generalization)")
args = parser.parse_args()

# ── 1. 克隆 / 更新仓库 ────────────────────────────────────────
banner("1/5  克隆 / 更新仓库")
if REPO_DIR.exists():
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

os.chdir(str(REPO_DIR))

# ── 2. 下载并解压数据集 ───────────────────────────────────────
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
    run(["rm", "-rf", str(REPO_DIR / "wildtrack"), str(REPO_DIR / "wiltrack")], check=False)

    print(f"[INFO] 从 Google Drive 文件夹下载 wildtrack (ID: {args.gdrive_id}) ...")
    run([sys.executable, "-m", "pip", "install", "-q", "gdown==5.2.0"])
    ret = run([
        sys.executable, "-m", "gdown", "--folder", "--fuzzy", "--remaining-ok",
        f"https://drive.google.com/drive/folders/{args.gdrive_id}",
        "-O", str(REPO_DIR),
    ], check=False)
    if ret != 0:
        print("[ERROR] gdown --folder 下载失败")
        sys.exit(1)

    # 诊断：打印 REPO_DIR 下所有内容（2 层深）
    print("[DEBUG] 下载后目录结构：")
    for p in sorted(REPO_DIR.rglob("*")):
        depth = len(p.relative_to(REPO_DIR).parts)
        if depth <= 2:
            print(f"  {'  ' * (depth-1)}{p.name}{'/' if p.is_dir() else ''}")

    # 递归查找 Image_subsets，确定数据实际落点
    found = list(REPO_DIR.rglob("Image_subsets"))
    if not found:
        print("[ERROR] 找不到 Image_subsets，Drive 文件夹结构异常")
        sys.exit(1)
    actual_root = found[0].parent  # Image_subsets 的父目录就是 wildtrack root
    if actual_root != data_root:
        if data_root.exists():
            import shutil; shutil.rmtree(str(data_root))
        actual_root.rename(data_root)
        print(f"[OK] 数据移动到：{data_root}")
    print(f"[OK] 数据就绪：{data_root}")

# ── 3. 验证数据集 ─────────────────────────────────────────────
banner("3/5  验证数据集结构")
for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
    p = data_root / subdir
    if p.exists():
        print(f"  [OK] {subdir}")
    else:
        print(f"  [ERROR] 缺少：{subdir}")
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
    "--max_frames",             str(args.max_frames),
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
    "--freeze_backbone_epochs", "0",
    "--bev_pos_weight",         str(args.bev_pos_weight),
    "--device",                 args.device,
    "--log_every",              "20",
]

print("命令：", " ".join(train_cmd))
ret = run(train_cmd, cwd=str(REPO_DIR), check=False)
if ret != 0:
    sys.exit(ret)

print(f"\n[OK] 训练完成")
print(f"模型：{REPO_DIR / 'outputs/train_multicam_mvdet_style_v3/model_final.pth'}")
