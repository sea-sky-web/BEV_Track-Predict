# AI Context — 2026-06-24

## 当前任务目标

**持续训练直到 BEV 俯视图中的行人预测可以和监督数据对齐。**

评判标准：`bev_prediction.png` 中的热力图高响应区域与 `bev_overlay.png` 中的 GT 行人位置在视觉上对齐，同时 MODA ≥ 0.3。

---

## 最新训练结果（Run 28070980141）

- **触发时间**：2026-06-24T02:28:45Z
- **配置**：epochs=10, bev_pos_weight=10.0, gpu=T4, max_frames=100, backbone=resnet18, fusion_mode=confidence_v2
- **训练结果**：全部 10 epoch 完成，model_final.pth 已下载

### 逐 Epoch 指标

| Epoch | loss   | bev_loss | raw_pos_mse | SNR   |
|-------|--------|----------|-------------|-------|
| 0     | 0.3839 | 0.3650   | 0.2462      | 0.252 |
| 1     | 0.1982 | 0.1952   | 0.1490      | 0.362 |
| 2     | 0.1633 | 0.1609   | 0.1340      | 0.419 |
| 3     | 0.1389 | 0.1366   | 0.1255      | 0.464 |
| 4     | 0.1220 | 0.1198   | 0.1163      | 0.487 |
| 5     | 0.1038 | 0.1017   | 0.1069      | 0.519 |
| 6     | 0.0929 | 0.0908   | 0.1005      | 0.535 |
| 7     | 0.0857 | 0.0836   | 0.0949      | 0.542 |
| 8     | 0.0809 | 0.0788   | 0.0910      | 0.548 |
| **9** | **0.0788** | **0.0767** | **0.0886** | **0.547** |

### 判断

- Loss 从 0.384 → 0.079（下降 79%），训练收敛正常
- SNR 全程为正（0.25 → 0.55），行人区域 logit 始终高于背景，模型在学习
- Epoch 8→9 SNR 几乎不变（0.548→0.547），**在 max_frames=100 配置下已接近饱和**
- MODA 未知（见下文问题说明）

---

## 当前工作流问题

### 问题 1：eval 结果无法下载（核心阻塞）

**现象**：eval 步骤运行 7.5 分钟后 exit code=0（成功），但 Colab session 在 VIZ 步骤开始时立刻 404/401，导致：
- `eval_results.json` 写入了 Colab，但未被下载
- `bev_prediction.png` / `bev_overlay.png` 未生成/未下载
- `[WARN] eval_results.json not found` 出现在日志中

**根因**：eval 完成后 Colab runtime 立刻被回收，session 消失。`colab exec` 的 stdout 里有 MODA 等数值，但 GitHub Actions 日志只捕获了命令回显，没有捕获 colab exec 的实际输出。

**修复方案（已确认）**：
在 eval 的 `colab exec` 块内，在 subprocess 调用结束后直接读取并打印 `eval_results.json` 内容到 stdout，这样 GA 日志里就能直接看到 MODA 数字，不依赖后续下载步骤。

```python
# 在 eval colab exec 块末尾追加：
import json, pathlib
p = pathlib.Path('/content/BEV_Track-Predict/outputs/eval_results.json')
if p.exists():
    r = json.loads(p.read_text())
    print('\n=== EVAL RESULTS (inline) ===')
    for k in ['det_moda','det_modp','det_precision','det_recall','det_f1',
              'det_best_threshold','det_moda_tp','det_moda_fp','det_moda_fn']:
        print(f'{k}: {r.get(k, "N/A")}')
    print('=== END EVAL RESULTS ===')
```

### 问题 2：max_frames=100 已饱和，需要更多数据

Epoch 8→9 SNR 不再增长，说明 100 帧的信息量已被充分利用。
**下一轮应将 max_frames 提升至 200 或 300**，同时可以增加 epochs 至 20。

### 问题 3：BEV 可视化从未成功生成

每一轮 bev_prediction.png 都没有被获取到，因为 viz 步骤一直在 eval 后 session 已死的情况下运行。
修复 eval 下载问题后，viz 步骤也需要在同一个 colab exec 块内执行，而不是依赖下一个 step 的 session 存活。

---

## 本轮对话已完成的工作

### 1. Harness 设计方案（已确认，待实现）

针对本研究项目定制的 harness，核心是研究迭代循环：`sanity → train → watch → inspect → 决策 → repeat`

**`scripts/harness.py` 命令设计**：

- `sanity [--data_root PATH]`：触发 Colab 之前的本地快速检查（30s）。有数据时跑 2帧1epoch 真实训练代码，无数据时只做 import + forward check
- `train [--epochs N] [--pos_weight W] [--gpu GPU]`：触发 GitHub Actions workflow dispatch
- `watch [--run_id N] [--poll SECS] [--out DIR]`：轮询直到完成，自动下载 artifacts，调用 inspect
- `inspect [--dir DIR]`：核心命令，读取 eval_results.json + metrics.csv，输出 ALIGNED/IMPROVING/STUCK/FAILED 四态判断及建议
- `history [--ai_runs DIR]`：从 ai_runs/ 读取历史，打印跨轮次 MODA/SNR 趋势表
- `loop [--epochs N] [--max_runs N] [--moda_target F]`：全自动化训练迭代

**inspect 判断逻辑**：

| 状态 | 条件 | 建议 |
|------|------|------|
| ALIGNED | MODA ≥ target | 完成 |
| IMPROVING | SNR > 0，MODA < target | 继续训练 / 加 epoch 或 max_frames |
| STUCK | SNR ≤ 0，loss 不降 | 调大 pos_weight 或 lr |
| FAILED | eval_results.json 缺失 | eval 步骤未运行，检查 workflow |

**注意**：旧版 `scripts/harness.py`（通用 SW 工程风格）已存在，需要完全替换为上述研究迭代风格。

### 2. 测试文件调整（已确认）

- `tests/test_trainer_harness.py`：**待删除**（合成数据测试，对研究无意义）
- `tests/test_loss.py`：**保留**（验证 pos_weight 梯度放大的真实约定）
- 其余已有测试：**保留**

### 3. 工作流文件状态

- `.github/workflows/colab-train.yml`：存在上述 eval session 死亡问题，**待修复**
- `.github/workflows/python-smoke.yml`：已更新包含 test_loss.py，但因 test_trainer_harness.py 待删除需再调整

---

## 下一步行动（优先级顺序）

### Step 1：修复 colab-train.yml（最高优先级）
在 eval colab exec 块内直接打印 eval_results.json，使 MODA 出现在 GA 日志中，不依赖后续下载。

### Step 2：实现 scripts/harness.py（研究版本）
按上述设计方案实现，替换现有的 harness.py。

### Step 3：删除 tests/test_trainer_harness.py，更新 python-smoke.yml

### Step 4：触发新一轮训练
配置：epochs=20, max_frames=200, bev_pos_weight=10.0, gpu=T4
观察 MODA 和 BEV 图是否对齐。

---

## 项目关键配置记录

```
架构：ResNet18Stride8Trunk → warp_perspective_torch → ConcatAttentionFusion → BEVHeadDilated
输出：BEV logits (B, 1, 360, 120)
损失：WeightedGaussianMSE，pos_weight=10.0
数据：Wildtrack 7相机，~400帧，max_frames=100（需提升至200+）
训练：adam lr=1e-4, cosine scheduler, 10 epochs
输出目录：outputs/train_multicam_mvdet_style_v3/
```

## GitHub Actions 运行记录

| Run ID | 结论 | 说明 |
|--------|------|------|
| 28029545309 | cancelled | 早期取消 |
| 28034425606 | cancelled | 早期取消 |
| 28038055791 | cancelled | 因 --timeout 不限制总时间导致挂起 67min 后手动取消 |
| 28043291879 | failure | 训练成功但 eval/viz 因 if: success() 被跳过 |
| **28070980141** | **success** | 训练+eval 均完成，但 eval 结果因 session 死亡未下载，MODA 未知 |
