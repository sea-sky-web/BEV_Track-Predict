---
name: gpu-priority
description: Training runs should default to A100 GPU, only downgrade to T4/L4 when A100 is unavailable
metadata:
  type: feedback
---

训练优先使用 A100 GPU，只有在 A100 不可用或受限时才降级到 L4 / T4。

**Why:** A100 训练速度远快于 T4（约 5-10x），用户不想在低算力上浪费时间。

**How to apply:** 触发 colab-train.yml 时 `-f gpu=A100`，如果 session 创建失败再改用 T4。
