# AI Context — 2026-06-25 (Update 3)

## 上一次结果

Run #47（GA run 28168973721）：concat + SGD lr=0.1 + OneCycleLR + AMP + grad_clip(10.0)

| 指标 | 值 |
|------|-----|
| MODA | N/A（eval 因 fusion_mode 不匹配崩溃） |
| 训练 SNR | -0.000（全 10 epoch） |
| pred_raw 范围 | [-0.031, 0.177] |
| 训练 loss | ~0.57 不下降 |

## 观察到的问题

### 问题 1：梯度爆炸导致训练坍缩

OneCycleLR warmup 阶段（step 0-600，lr 从 0.004 → 0.1）：
- Step 0-400: 正常学习（SNR=0.1-0.3，pred_raw 有合理范围）
- Step 420 (lr≈0.08): 首次爆炸（loss=7.58）
- Step 440 (lr≈0.08): 灾难性爆炸（loss=3308, pred_raw 飙至 154）
- Step 480: 再次爆炸（loss=4154, pred_raw=-993）
- Step 620+ (lr=0.1 peak): 模型权重已被破坏，永久坍缩为零输出

**根因**: concat 模式下 BEV head 第一层 conv 为 (512, 3586, 3, 3) = 16.5M 参数。
高 lr + AMP 的 loss scaling 双重放大梯度，max_norm=10.0 的 clip 不够紧。

### 问题 2：eval 因 fusion_mode 不匹配崩溃

config.py 中 DEFAULT_FUSION_MODE 仍为 "confidence_v2"，但 checkpoint 是 concat 训练的。
Eval 构建了 514 通道输入的模型，无法加载 3586 通道的权重。

### 问题 3：visualize_prediction.py 导入错误

`from src.model import MultiviewDetector` — 模块名和类名都不对。

## 本轮修复

### 修复 1：降低 lr 并收紧梯度裁剪

- `config.py`: DEFAULT_MAX_LR / DEFAULT_LR_INIT: 0.1 → 0.01
- `colab_train.py`: --lr_init 0.1 → 0.01
- `trainer.py`: grad_clip max_norm: 10.0 → 1.0

**理由**: concat 的 3586 通道输入使梯度量级约为 confidence_v2（514 通道）的 7 倍。
MVDet 原始 lr=0.1 适配全量 ~400 帧（4000 steps），我们用 100 帧（1000 steps），
warmup 斜率更陡。降到 0.01 + clip=1.0 应当防止爆炸同时保持学习能力。

### 修复 2：DEFAULT_FUSION_MODE → "concat"

- `config.py`: DEFAULT_FUSION_MODE = "concat"
- `colab_train.py` eval 命令: 显式传 --fusion_mode concat --backbone resnet18
- `colab-train.yml` eval 步骤: 同上

### 修复 3：重写 visualize_prediction.py

使用当前 API（create_model, create_wildtrack_dataset），支持 --fusion_mode 参数。

### 修复 4：README 同步

训练/评估示例命令更新为 concat + SGD lr=0.01 + OneCycleLR。

## 预期

下一次 Colab run 应当：
- 训练 loss 持续下降，不出现梯度爆炸
- SNR 在训练中逐步上升
- eval 正常运行，产出 MODA/MODP/Precision/Recall/F1
- 可视化脚本正常生成 bev_prediction.png

如果 lr=0.01 仍无法收敛（loss 不降），下一步尝试 lr=0.05 + max_norm=1.0。
如果收敛但 MODA 仍低于 0.30，考虑增加 max_frames 到 200-400。
