#!/usr/bin/env python3
"""在 Colab 上读取 MVDet 官方代码的训练配置，不需要 GPU。"""
import os
import subprocess
import sys

# Clone MVDet if not present
mvdet_dir = "/content/MVDet"
if not os.path.exists(mvdet_dir):
    subprocess.run(["git", "clone", "https://github.com/hou-yz/MVDet.git", mvdet_dir], check=True)

print("=" * 60)
print("  MVDet 训练配置分析")
print("=" * 60)

# 1. main.py — argparse defaults
print("\n--- main.py argparse defaults ---")
main_py = os.path.join(mvdet_dir, "main.py")
if os.path.exists(main_py):
    with open(main_py, encoding="utf-8") as f:
        for line in f:
            if any(k in line.lower() for k in ["add_argument", "lr", "epoch", "momentum", "weight_decay",
                                                  "batch", "augment", "flip", "scheduler", "split",
                                                  "train", "test", "optim"]):
                print(line.rstrip())
else:
    print("[WARN] main.py not found")

# 2. trainer.py — optimizer/scheduler creation
print("\n--- trainer.py optimizer/scheduler ---")
for fname in ["multiview_detector/trainer.py", "multiview_detector/models/trainer.py"]:
    fpath = os.path.join(mvdet_dir, fname)
    if os.path.exists(fpath):
        print(f"\n[FILE] {fname}")
        with open(fpath, encoding="utf-8") as f:
            content = f.read()
            print(content)
        break

# 3. dataset split — frameDataset or Wildtrack
print("\n--- frameDataset train/test split ---")
for fname in ["multiview_detector/datasets/frameDataset.py",
              "multiview_detector/datasets/Wildtrack.py"]:
    fpath = os.path.join(mvdet_dir, fname)
    if os.path.exists(fpath):
        print(f"\n[FILE] {fname}")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if any(k in line.lower() for k in ["train", "test", "split", "frame", "augment",
                                                      "flip", "transform", "ratio", "len(", "__len__"]):
                    print(line.rstrip())

# 4. augmentation — any transforms
print("\n--- augmentation / transforms ---")
for root, dirs, files in os.walk(mvdet_dir):
    dirs[:] = [d for d in dirs if d not in {".git", "data", "logs", "outputs"}]
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if any(k in line.lower() for k in ["randomhorizontalflip", "colorjitter",
                                                         "randomcrop", "augment"]):
                        print(f"  {fpath}:{i}: {line.rstrip()}")

print("\n--- END ---")
sys.exit(0)
