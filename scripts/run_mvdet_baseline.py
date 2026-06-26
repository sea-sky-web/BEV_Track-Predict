#!/usr/bin/env python3
"""
在 Colab 上运行 MVDet 官方代码，获取 WildTrack 基线 MODA。

用法（GitHub Actions 通过 colab exec 运行）：
    cat scripts/run_mvdet_baseline.py | colab exec -s bev-train --timeout 7200
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

MVDET_REPO = "https://github.com/hou-yz/MVDet.git"
MVDET_DIR = Path("/content/MVDet")
WILDTRACK_SRC = Path("/content/BEV_Track-Predict/wildtrack")
WILDTRACK_DST = Path(os.path.expanduser("~/Data/Wildtrack"))


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


# ── 1. Clone MVDet ─────────────────────────────────────────
banner("1/5  Clone MVDet 官方仓库")
if MVDET_DIR.exists():
    run(["git", "-C", str(MVDET_DIR), "pull", "--ff-only"], check=False)
else:
    run(["git", "clone", "--depth", "1", MVDET_REPO, str(MVDET_DIR)])

# ── 2. 安装依赖 ────────────────────────────────────────────
banner("2/5  安装 MVDet 依赖")
run([sys.executable, "-m", "pip", "install", "-q",
     "kornia", "scipy", "opencv-python", "matplotlib", "pillow", "tqdm"])

# ── 3. 链接 WildTrack 数据 ──────────────────────────────────
banner("3/5  链接 WildTrack 数据")
if not WILDTRACK_SRC.exists():
    print(f"[ERROR] WildTrack 数据不存在：{WILDTRACK_SRC}")
    print("[INFO] 需要先运行 colab_train.py 下载数据")
    sys.exit(1)

WILDTRACK_DST.parent.mkdir(parents=True, exist_ok=True)
if WILDTRACK_DST.is_symlink() or WILDTRACK_DST.exists():
    run(["rm", "-rf", str(WILDTRACK_DST)])
WILDTRACK_DST.symlink_to(WILDTRACK_SRC)
print(f"[OK] {WILDTRACK_DST} -> {WILDTRACK_SRC}")

# 验证数据结构
for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
    p = WILDTRACK_DST / subdir
    assert p.exists(), f"缺少 {subdir}: {p}"
    print(f"  [OK] {subdir}")

# ── 4. 适配 MVDet 为单 GPU ──────────────────────────────────
banner("4/5  适配 MVDet 为单 GPU")

# MVDet 的 PerspTransDetector 硬编码了 cuda:0 和 cuda:1
# 需要 patch 为单 GPU
model_file = MVDET_DIR / "multiview_detector" / "models" / "persp_trans_detector.py"
model_code = model_file.read_text()

if "'cuda:1'" in model_code:
    patched = model_code.replace("'cuda:1'", "'cuda:0'")
    model_file.write_text(patched)
    print("[OK] 已 patch PerspTransDetector: cuda:1 -> cuda:0")
else:
    print("[OK] 已经是单 GPU 模式")

# MVDet 的 evaluate.py 默认用 MATLAB，需确保 fallback 到 python eval
eval_file = MVDET_DIR / "multiview_detector" / "evaluation" / "evaluate.py"
if eval_file.exists():
    print(f"[OK] evaluate.py 存在，将使用 python fallback")

# ── 5. 运行 MVDet 训练 ──────────────────────────────────────
banner("5/5  运行 MVDet 训练 (10 epochs)")

train_cmd = [
    sys.executable, "main.py",
    "-d", "wildtrack",
    "--arch", "resnet18",
    "--epochs", "10",
    "--lr", "0.1",
    "--batch_size", "1",
    "--momentum", "0.5",
    "--weight_decay", "5e-4",
    "--log_interval", "50",
    "--seed", "1",
]
print("命令：", " ".join(train_cmd))
ret = run(train_cmd, cwd=str(MVDET_DIR), check=False)
print(f"\n[TRAIN] exit code: {ret}", flush=True)

# ── 收集结果 ────────────────────────────────────────────────
banner("结果摘要")

# 查找最新 log 目录
log_base = MVDET_DIR / "logs" / "wildtrack_frame" / "default"
if log_base.exists():
    log_dirs = sorted(log_base.iterdir())
    if log_dirs:
        latest_log = log_dirs[-1]
        log_file = latest_log / "log.txt"
        if log_file.exists():
            log_text = log_file.read_text()
            # 提取最后的 moda 行
            moda_lines = [l for l in log_text.splitlines() if "moda:" in l.lower()]
            if moda_lines:
                print("\n=== MVDet BASELINE RESULTS ===")
                for line in moda_lines[-3:]:
                    print(line)
                print("=== END MVDet BASELINE ===")
            else:
                print("[WARN] 未找到 moda 输出")
                # 打印最后 20 行
                lines = log_text.splitlines()
                for line in lines[-20:]:
                    print(line)
        
        # 保存 checkpoint 路径
        ckpt = latest_log / "MultiviewDetector.pth"
        if ckpt.exists():
            print(f"\n[OK] MVDet checkpoint: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")
else:
    print("[WARN] 未找到 MVDet 日志目录")

print("\n[DONE] MVDet baseline 运行完成", flush=True)
