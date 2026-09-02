# AI Iteration Context

## 1. Iteration ID

20260902_082709

## 2. Previous Iteration

ai_runs/20260629_180955/（latest_run.txt 指向的最后一次训练迭代）

## 3. Previous Metrics Summary

| 指标 | 数值 | 来源 |
|------|:----:|------|
| MODA | 0.8950 | docs/research_progress.md §3.1 |
| MODP | 0.7778 | 同上 |
| Precision | 0.9301 | 同上 |
| Recall | 0.9223 | 同上 |
| F1 | 0.9262 | 同上 |
| MODA 复核 | 0.8634 | docs/research_progress.md §10.7（cuDNN 差异） |
| advection AUPRC | 0.7645 | §10.5 |
| const-vel ADE | 0.1555 m | §10.9 |
| ConvLSTM AUPRC | 0.0301 | §10.6 |

**注意**：以上所有数字经本次审计确认存在协议缺陷，不可直接用于论文。详见 `docs/paper_readiness_audit.md`。

## 4. Observed Problem

`docs/research_progress.md` 中的头条数字存在多处协议和代码层面的缺陷，包括：

1. 检测阈值/NMS 半径在测试集上通过 132 组网格扫描取 argmax（`src/evaluate_main.py:505,723`）
2. 评估 GT 来自 `adaptive_max_pool2d` 池化热力图而非原始标注（`src/evaluate_main.py:294`）
3. MODA（Hungarian + 0.5 m）和 P/R/F1（贪心 + 0.3 m）使用两套不同匹配器
4. 推理延迟/FPS 在未截断版 MobileNet-V2（8.0M）上测量，非产出 MODA 的截断版（5.7M）
5. 单次运行、无种子、无误差棒
6. Module 2 常速度基线泄漏未来（中心差分），AUPRC 阈值网格依赖预测值量级
7. `fig9_tracking.png` 是合成随机游走数据配真实图注
8. offset head 训练目标恒为 0，batch>1 崩溃
9. 仓库自我审查文档（`second_stage_innovation_review.md`）的 P0.1/P0.5 结论未回流到代码

## 5. Improvement Hypothesis

N/A（本次迭代为审计，不涉及模型改进）

## 6. Changes Made

Changed files:
- `docs/paper_readiness_audit.md`（新建）：A1–A12 代码缺陷 + B1–B6 协议问题 + C1–C3 Module 2 问题 + D1–D9 论文未完成项
- `docs/active_plan.md`（覆盖）：审计驱动的修复计划，P0→P3 优先级排序
- `docs/LESSONS.md`（追加）：B12–B17 实验教训 + Part A 方法论规则 #9（全仓库协议审计）
- `docs/daily-log.md`（顶部插入）：2026-09-02 审计条目
- `docs/research_progress.md`（顶部加声明）：结果有效性声明 + §9 扩充待完成实验
- `ai_runs/20260902_082709/`（新建）：本迭代记录
- `.remember/now.md`（更新）：当前状态

变更类型：documentation change（无代码变更）

## 7. Training Configuration

Training not performed in this iteration.

## 8. Evaluation Configuration

Evaluation not performed in this iteration.

测试套件实跑：`PYTHONPATH=src python3 -m pytest tests -q` → 1 failed, 136 passed in 7.77s
（失败测试为 `test_augmentation.py::test_view_coherent_hflip_flips_images_bev_and_aux_labels`，根因浮点舍入 ~1.9e-9，`allclose` 通过）

## 9. Current Metrics

N/A（本次迭代为审计，无模型训练或评估）

## 10. Result Interpretation

N/A

本次审计确认：当前 `research_progress.md` 表格里没有一个数字可以直接进论文。距离可投稿状态，保守估计需要 P0 修复 + 全量重跑 → P1 消融补全 → 写作 → P2 竞争力提升。

## 11. Next Iteration Recommendation

Next action:
按 P0 优先级修复后全量重跑。第一步：D2（fig9 替换或删除）。

Reason:
当前所有头条数字均因协议缺陷而无效，修复后数字会变化（方向已知，幅度未知），后续所有消融和写作必须基于修复后的数字。

Expected validation:
一张新的主表，固定超参（threshold=0.400, NMS=6.0，在验证集上选定），世界坐标 GT，统一贪心匹配 + 0.5 m，3 种子 mean ± std。

## 12. Do Not Do Next

- Do not start writing the paper — current numbers are all invalid
- Do not propose new model improvements — fix protocol before any new experiments
- Do not modify existing numbers in `research_progress.md` — preserve history, new numbers go into new iterations
- Do not introduce BEVFormer/PETR/LSS/DETR3D or other large BEV frameworks
- Do not add tracking/ReID/trajectory prediction modules beyond what already exists
- Do not delete any `ai_runs/` historical experiment records
- Do not commit code changes without user review (AGENTS.md §16 red line)