# 训练坍缩修复计划

## Context

训练 10 epoch 后模型完全不可用：`aux_pos_mse ≈ 1.0`，`pred_raw` 均值 ≈ -3.7（sigmoid ≈ 0.024），模型学会"全部预测 0"。

通过对比本地存档的 MVDet 原始实现（`archive/legacy/training_prototypes/final_train_multicam_mvdet_style.py`）与当前 `src/` 代码，定位到 **3 个关键偏差**：

| 属性 | MVDet 原始 | 当前实现 | 严重性 |
|------|-----------|---------|--------|
| Loss 输入 | **原始 logits**（无 sigmoid） | `torch.sigmoid()` 后再传入 MSE | **致命** — 坍缩主因 |
| BEV 高斯核 | sigma=5.0, ksize=41 | sigma=2.5, ksize=11 | 高 — 正样本信号过窄 |
| 图像高斯核 | sigma=2.5, ksize=21 | sigma=2.0, ksize=11 | 中 |

## 坍缩机制

1. Sigmoid 初始输出 ≈ 0.5，但高斯模糊后的 GT 峰值仅 0.1-0.4
2. MSE 梯度将 logits 推向极负值 → sigmoid → 0
3. 20 步内模型发现"全预测 0"是 MSE 最小解
4. 模型再也无法恢复

## 修复策略：3 步渐进式，每步验证

### Step 1：移除 loss 前的 sigmoid（对齐 MVDet）— 最关键

**文件**：`src/trainer.py`

**修改**：
- `train_epoch()` 中：将原始 logits 直接传给 `bev_criterion` 和 `img_criterion`
- sigmoid 仅保留用于监控指标（pos_mse、aux_pos_mse）和可视化
- `validate()` 同步修改

**验证**：
1. 本地 `pytest tests/` 通过
2. Actions 训练 → 观察 step 0-100 的 `pos_mse` 趋势：修复后应持续下降
3. `pred_raw` mean 不再单向暴跌

### Step 2：对齐高斯核参数

**文件**：`src/config.py`

**修改**：`MAP_KSIZE=41, MAP_SIGMA=5.0, IMG_KSIZE=21, IMG_SIGMA=2.5`

**验证**：pytest 通过 + 训练 loss 更平滑

### Step 3：freeze_backbone_epochs 3 → 0

**文件**：`src/config.py`

**验证**：对比 Step 2 结果

## 每步落盘规则

1. 本地 pytest 通过 → commit
2. Actions 训练 → 关键指标记录到 `docs/fix_training_collapse.md`
3. 指标恶化 → 立即 revert

---

## Step 1 验证结果（2026-06-22）

**Commit**: `43a64e8` — "fix(critical): pass raw logits to GaussianMSE, not sigmoid output"
**Actions Run**: https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/27943641545
**GPU**: A100, 10 epochs

### 关键指标对比

| 指标 | 修复前（坍缩） | Step 1 修复后 | 判定 |
|------|----------------|---------------|------|
| pos_mse step 0 | 0.196 | 0.196 | 起点一致 |
| pos_mse step 20 | **0.58**（暴涨） | **0.249**（稳定） | ✅ 不再坍缩 |
| pos_mse epoch 9 | ~1.0 | **0.14-0.21** | ✅ 持续下降 |
| aux_pos_mse | 恒定 ~1.0 | **0.218-0.225** | ✅ 大幅改善 |
| pred_raw mean | **-3.7**（全零输出） | **0.01-0.03** | ✅ 模型在学习 |
| pred_raw max | ~0 | **0.5-1.1** | ✅ 有正向激活 |
| loss 趋势 | 不收敛 | **0.022→0.008** | ✅ 稳定下降 |

### Epoch 级别 loss 趋势

| Epoch | Loss | BEV | IMG |
|-------|------|-----|-----|
| 0 | 0.0221 | 0.0186 | 0.0035 |
| 1 | 0.0119 | 0.0104 | 0.0015 |
| 2 | 0.0114 | 0.0100 | 0.0014 |
| 6 | 0.0092 | 0.0080 | 0.0012 |
| 7 | 0.0088 | 0.0076 | 0.0012 |
| 8 | 0.0085 | 0.0073 | 0.0012 |
| 9 | 0.0084 | 0.0072 | 0.0012 |

### 结论

**Step 1 修复成功**。模型不再坍缩，持续学习。但 pos_mse 最低仅到 ~0.14，说明高斯核参数（sigma=2.5, ksize=11）生成的正样本信号太窄，模型难以精确对齐峰值。继续 Step 2。

---

## Step 2 验证结果（2026-06-22）

**Commit**: `ec891e7` — "fix(step2): align Gaussian kernel params to MVDet"
**Actions Run**: https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/27954744684
**GPU**: T4, 10 epochs

**修改内容**：`src/config.py` — MAP_KSIZE=11→41, MAP_SIGMA=2.5→5.0, IMG_KSIZE=11→21, IMG_SIGMA=2.0→2.5

### 关键指标对比（vs Step 1）

| 指标 | Step 1 | Step 2 | 判定 |
|------|--------|--------|------|
| pos_mse 最佳 | ~0.14 | **~0.094** | ✅ 下降 33% |
| aux_pos_mse | 0.218-0.225 | **0.210-0.213** | ✅ 改善 |
| pred_raw max | 0.5-1.1 | **1.5-2.0** | ✅ 激活翻倍 |
| pred_raw mean | 0.01-0.03 | **0.05-0.13** | ✅ 更自信 |
| loss epoch 9 | 0.008 | 0.021 | 预期（GT 峰值更高） |

### Epoch 级别 loss 趋势

| Epoch | Loss | BEV | IMG |
|-------|------|-----|-----|
| 0 | 0.0566 | 0.0528 | 0.0038 |
| 1 | 0.0428 | 0.0406 | 0.0022 |
| 2 | 0.0395 | 0.0374 | 0.0021 |
| 7 | 0.0239 | 0.0219 | 0.0020 |
| 8 | 0.0223 | 0.0203 | 0.0019 |
| 9 | 0.0215 | 0.0195 | 0.0019 |

### 结论

**Step 2 修复成功**。宽高斯核让正样本信号更充分，pos_mse 从 0.14 进一步降到 0.094，pred_raw max 从 1.1 提升到 2.0。Loss 绝对值更高是因为 GT 峰值从 ~0.4 提升到 ~1.0（sigma 从 2.5→5.0）。继续 Step 3。

---

## 不做的事

- 不改 loss 类型（保持 GaussianMSE）
- 不改优化器（保持 Adam + Cosine）
- 不改模型架构、数据集或增强策略
