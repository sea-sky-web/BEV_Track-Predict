# Plan: 评估指标 Action（独立分支 + 独立 workflow）

## Context

训练坍缩修复的 3 步计划正在执行（Step 3 进行中），但目前只有训练 loss 指标（pos_mse, aux_pos_mse），没有检测级评估指标（MODA/MODP/Precision/Recall/F1）。需要构建一个独立的评估 workflow，可以对任意训练产出的模型运行标准检测评估。

**好消息**：评估核心代码已经存在：
- `src/evaluate_main.py` — 完整的离线评估入口，支持 `--report_detection` 输出 MODA/MODP
- `src/metrics.py` — `compute_moda_modp` + `aggregate_metrics`（匈牙利匹配）
- `tests/test_metrics.py` — 已有单元测试

**缺失的**：没有 Colab 上运行评估的 bootstrap 脚本和 GitHub Actions workflow。

## 实施方案

### 1. 创建分支 `feat/eval-workflow`

从 main 分支创建。

### 2. 新建 `scripts/colab_eval.py`

仿照 `scripts/colab_train.py` 的结构，用于在 Colab 上执行评估：
- clone 仓库 + gdown 下载数据集（复用 colab_train.py 的逻辑）
- 接收参数：`--model_path`（Colab 上模型路径）
- 调用 `src/evaluate_main.py --report_detection --model_path <path>`
- 同时运行 `scripts/visualize_prediction.py` 生成可视化
- 输出文件：`eval_results.json`、`bev_prediction.png`、`bev_overlay.png`

### 3. 新建 `.github/workflows/colab-eval.yml`

`workflow_dispatch` 触发，独立于训练 workflow：
- 输入参数：
  - `run_number`：训练 run 编号（用于下载 artifact）
  - `gpu`：默认 T4（评估不需要高端 GPU）
- 步骤：
  1. Checkout
  2. Install colab-cli + 恢复凭证
  3. Download 训练 artifact（model_final.pth）
  4. Create Colab session（T4）
  5. Upload 模型到 Colab
  6. `colab exec` 运行 colab_eval.py
  7. Download 评估结果 + 可视化
  8. Upload artifacts（eval_results.json, bev_prediction.png, bev_overlay.png）
  9. Stop session

### 4. 关键文件

| 文件 | 操作 |
|------|------|
| `scripts/colab_eval.py` | **新建** — Colab 评估 bootstrap |
| `.github/workflows/colab-eval.yml` | **新建** — 评估 workflow |
| 已有文件无需修改 | `src/evaluate_main.py`, `src/metrics.py` 直接复用 |

### 5. 评估输出格式

`eval_results.json` 包含：
```json
{
  "best_threshold": 0.35,
  "moda": 0.72,
  "modp": 0.68,
  "precision": 0.85,
  "recall": 0.78,
  "f1": 0.81,
  "tp": 312,
  "fp": 55,
  "fn": 88
}
```

## 验证

1. 推送到 `feat/eval-workflow` 分支
2. 等 Step 3 训练完成后，用其 artifact 触发评估 workflow
3. 检查输出的 MODA/MODP 数值和可视化图
