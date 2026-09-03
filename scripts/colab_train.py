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
parser.add_argument("--bev_pos_weight", type=float, default=1.0,
                    help="BEV 正样本损失权重")
parser.add_argument("--device", default="cuda")
parser.add_argument("--max_frames", type=int, default=320,
                    help="Max training frames (train split = 320, val = 320-359, test = 360-399)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--loss_type", default="mse", choices=["mse", "focal"],
                    help="BEV loss: mse (MVDet baseline) or focal (CenterNet-style)")
parser.add_argument("--offset_weight", type=float, default=0.0,
                    help="Offset L1 loss weight (0.0 = disable offset training)")
parser.add_argument("--fusion_mode", default="confidence_v2",
                    choices=["concat", "confidence_v1", "confidence_v2", "geo_confidence_v1"],
                    help="BEV fusion mode (baseline = confidence_v2)")
parser.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v2"],
                    help="Backbone network (resnet18 = MVDet baseline)")
parser.add_argument("--branch", default=None,
                    help="Git branch to checkout after clone (default: repo default branch)")
args = parser.parse_args()

# ── 1. 克隆 / 更新仓库 ────────────────────────────────────────
banner("1/6  克隆 / 更新仓库")
if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
else:
    if REPO_DIR.exists():
        # Directory exists but not a git repo (e.g. created by checkpoint upload)
        # Preserve outputs/ then clone fresh
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

if args.branch:
    run(["git", "-C", str(REPO_DIR), "fetch", "origin", args.branch], check=False)
    run(["git", "-C", str(REPO_DIR), "checkout", args.branch])
    run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

os.chdir(str(REPO_DIR))

# ── 2. 下载并解压数据集 ───────────────────────────────────────
banner("2/6  数据集准备")
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
        print("[WARN] gdown --folder 下载失败，尝试 fallback 路径 ...")
        fallback_paths = [
            Path("/content/wildtrack"),
            Path("/content/BEV_Track-Predict/wildtrack"),
            Path("/root/wildtrack"),
        ]
        found_fallback = False
        for fb in fallback_paths:
            if fb.is_dir() and (fb / "Image_subsets").exists():
                import shutil
                if data_root.exists():
                    shutil.rmtree(str(data_root))
                shutil.copytree(str(fb), str(data_root))
                print(f"[OK] 使用 fallback 数据：{fb}")
                found_fallback = True
                break
        if not found_fallback:
            print("[ERROR] gdown 下载失败且无可用 fallback 数据")
            sys.exit(1)

    # Drive 文件夹可能直接包含 wildtrack.zip — 解压
    zip_path = REPO_DIR / "wildtrack.zip"
    if zip_path.exists():
        print(f"[INFO] 发现 wildtrack.zip ({zip_path.stat().st_size / 1e9:.2f} GB)，解压中 ...")
        run(["unzip", "-q", str(zip_path), "-d", str(REPO_DIR)])
        zip_path.unlink()
        print("[OK] 解压完成")

    # 递归查找 Image_subsets，确定数据实际落点
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

# ── 3. 验证数据集 ─────────────────────────────────────────────
banner("3/6  验证数据集结构")
for subdir in ["Image_subsets", "calibrations", "annotations_positions"]:
    p = data_root / subdir
    if p.exists():
        print(f"  [OK] {subdir}")
    else:
        print(f"  [ERROR] 缺少：{subdir}")
        run(["ls", "-la", str(REPO_DIR)], check=False)
        sys.exit(1)

# ── 4. 安装依赖 ───────────────────────────────────────────────
banner("4/6  安装 Python 依赖")
run([sys.executable, "-m", "pip", "install", "-q",
     "-r", str(REPO_DIR / "requirements.txt")])

# ── 5. 训练 ──────────────────────────────────────────────────
if args.epochs > 0:
    banner("5/6  开始训练")

    train_cmd = [
    sys.executable, "scripts/train_main.py",
    "--data_root",              str(data_root),
    "--views",                  "0,1,2,3,4,5,6",
    "--epochs",                 str(args.epochs),
    "--batch",                  "1",
    "--pretrained",             "true",
    "--backbone",               args.backbone,
    "--fusion_mode",            args.fusion_mode,
    "--augment",                "false",
    "--alpha",                  "1.0",
    "--optimizer",              "sgd",
    "--scheduler",              "onecycle",
    "--lr_init",                "0.1",
    "--max_frames",             str(min(args.max_frames, 320) if args.max_frames > 0 else 320),
    "--seed",                   str(args.seed),
    "--weight_decay",           "0.0005",
    "--freeze_backbone_epochs", "0",
    "--bev_pos_weight",         str(args.bev_pos_weight),
    "--device",                 args.device,
    "--log_every",              "20",
    "--loss_type",              args.loss_type,
    "--offset_weight",          str(args.offset_weight),
]

    print("命令：", " ".join(train_cmd))
    ret = run(train_cmd, cwd=str(REPO_DIR), check=False)
    if ret != 0:
        sys.exit(ret)

    print(f"\n[OK] 训练完成")
else:
    banner("5/6  跳过训练 (epochs=0, eval-only 模式)")

# ── 6. 评估 + 可视化（同一 colab exec，避免 session 死亡）────────
banner("6/7  验证集超参选择 (frames 320-359)")

model_path = REPO_DIR / "outputs/train_multicam_mvdet_style_v3/model_final.pth"
val_out    = REPO_DIR / "outputs/val_results.json"

val_cmd = [
    sys.executable, "src/evaluate_main.py",
    "--data_root",       str(data_root),
    "--model_path",      str(model_path),
    "--device",          args.device,
    "--report_detection",
    "--metrics_out",     str(val_out),
    "--views",           "0,1,2,3,4,5,6",
    "--fusion_mode",     args.fusion_mode,
    "--backbone",        args.backbone,
    "--frame_start",     "320",
    "--max_frames",      "40",
    "--det_thresholds=-0.50,-0.25,-0.10,0.00,0.05,0.10,0.15,0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50,0.55,0.60",
    "--det_min_distances=3.0,4.0,5.0,6.0,7.0,8.0",
    "--loss_type",       args.loss_type,
]
if args.offset_weight > 0:
    val_cmd.append("--use_offset")
print("命令：", " ".join(val_cmd))
val_ret = run(val_cmd, cwd=str(REPO_DIR), check=False)
print(f"\n[VAL] exit code: {val_ret}", flush=True)

best_thr = "0.400"
best_nms = "6.0"
if val_out.exists():
    vr = json.loads(val_out.read_text())
    best_thr = str(vr.get("det_best_threshold", 0.400))
    best_nms = str(vr.get("det_best_nms_radius", 6.0))
    print(f"\n[VAL] Best on validation: threshold={best_thr}, NMS={best_nms}")
    print(f"[VAL] Val MODA={vr.get('det_moda', 'N/A')}, F1={vr.get('det_f1', 'N/A')}")
else:
    print("[WARN] val_results.json not found, using defaults threshold=0.400, NMS=6.0")

banner("7/7  测试集评估 (frames 360-399, fixed hyperparams)")

eval_out = REPO_DIR / "outputs/eval_results.json"

eval_cmd = [
    sys.executable, "src/evaluate_main.py",
    "--data_root",       str(data_root),
    "--model_path",      str(model_path),
    "--device",          args.device,
    "--report_detection",
    "--metrics_out",     str(eval_out),
    "--views",           "0,1,2,3,4,5,6",
    "--fusion_mode",     args.fusion_mode,
    "--backbone",        args.backbone,
    "--frame_start",     "360",
    "--max_frames",      "40",
    f"--det_thresholds={best_thr}",
    f"--det_min_distances={best_nms}",
    "--loss_type",       args.loss_type,
]
if args.offset_weight > 0:
    eval_cmd.append("--use_offset")
print("命令：", " ".join(eval_cmd))
eval_ret = run(eval_cmd, cwd=str(REPO_DIR), check=False)
print(f"\n[EVAL] exit code: {eval_ret}", flush=True)

if eval_out.exists():
    r = json.loads(eval_out.read_text())
    print("\n=== TEST RESULTS (fixed hyperparams from val) ===")
    print(f"threshold={best_thr}, NMS={best_nms}")
    for k in ["det_moda", "det_modp", "det_precision", "det_recall", "det_f1",
              "det_moda_tp", "det_moda_fp", "det_moda_fn"]:
        print(f"{k}: {r.get(k, 'N/A')}")
    print("=== END TEST RESULTS ===", flush=True)
else:
    print("[WARN] eval_results.json not found", flush=True)

# 可视化（单帧）
viz_dir = str(REPO_DIR / "outputs/visualization")
viz_cmd = [
    sys.executable, "scripts/visualize_prediction.py",
    "--model_path",  str(model_path),
    "--data_root",   str(data_root),
    "--output",      viz_dir,
    "--frame",       "0",
]
run(viz_cmd, cwd=str(REPO_DIR), check=False)

print("[OK] 全部完成", flush=True)
