#!/usr/bin/env python3
"""
最小化模拟脚本：验证 colab exec 运行期间能否被 colab download 并发下载文件。

不涉及真实训练/模型代码，只生成一个假的 "checkpoint" 文件（随机 tensor），
模拟 model_final.pth 每隔一段时间更新一次的行为，持续运行一段时间，
让 GitHub Actions workflow 侧的周期性 colab download 有机会验证是否成功。

用法（Colab exec 中运行）：
    python3 scripts/fake_training_sim.py --duration 300 --interval 20
"""
import argparse
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=300, help="总模拟时长（秒）")
parser.add_argument("--interval", type=int, default=20, help="每次更新 checkpoint 的间隔（秒）")
parser.add_argument("--output", type=str, default="/content/fake_outputs")
args = parser.parse_args()

out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = out_dir / "model_final.pth"

start = time.time()
step = 0
while time.time() - start < args.duration:
    step += 1
    fake_state_dict = {
        "fake_layer.weight": torch.randn(64, 64),
        "fake_layer.bias": torch.randn(64),
        "step": torch.tensor(step),
        "timestamp": torch.tensor(time.time()),
    }
    torch.save(fake_state_dict, ckpt_path)
    elapsed = time.time() - start
    print(f"[SIM] step={step} elapsed={elapsed:.1f}s saved={ckpt_path} size={ckpt_path.stat().st_size}B", flush=True)
    time.sleep(args.interval)

print(f"[SIM] Done. Total steps={step}, final file={ckpt_path}", flush=True)
