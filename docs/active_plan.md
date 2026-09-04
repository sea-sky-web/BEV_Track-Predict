# Active Plan — 当前迭代

> 最后更新：2026-09-03
> 上一迭代：`docs/paper_readiness_audit.md`（2026-09-02 全仓库审计）
> 上一迭代结果：确认 A1–A12 代码缺陷 + B1–B6 协议问题 + C1–C3 Module 2 问题 + D1–D9 论文未完成项
> 协议依据：`docs/experiment_iteration_protocol.md` §11（唯一主要下一步动作）

---

## 当前状态：P0 修复完成，首轮重跑进行中

### 审计摘要

`docs/paper_readiness_audit.md` 已对全仓库完成逐文件审计（137 测试实跑），确认：

- **代码缺陷**：12 项（A1–A12），其中 A1（offset 训到 0）、A2（B>1 崩溃）、A3（池化 GT）为 correctness bug
- **协议/结论有效性**：6 项（B1–B6），其中 B1（测试集调参）、B2（方差与增益同量级）为致命问题
- **Module 2**：3 项（C1–C3），其中 C1（常速度泄未来）和 C3（AUPRC 尺度偏倚）可能导致核心结论反转
- **论文未完成**：9 项（D1–D9），D2（fig9 合成数据）为学术诚信问题

### 当前头条数字的有效性

| 数字 | 来源 | 可进论文？ | 原因 |
|------|------|:---:|------|
| MODA 0.8950 | `research_progress.md` | ❌ | B1（测试集 132 组超参 argmax）+ A3（池化 GT）+ A4（双匹配器） |
| P 0.9301 / R 0.9223 | 同上 | ❌ | A4（贪心 0.3 m 匹配） |
| 参数量 5.7M / 32.7M | 同上 | ⚠️ | 需验证 5.7M 截断版实际参数量 + 32.7M 复现版参数量 |
| FPS 0.96 / 延迟 1044ms | 同上 | ❌ | B5（在 8.0M 未截断版上测的） |
| ADE 0.1555 (const-vel) | 同上 | ❌ | C1（泄漏未来） |
| ConvLSTM AUPRC 0.0301 | 同上 | ❌ | C3（AUPRC 尺度偏倚） |
| fig9_tracking.png | `generate_paper_figures.py` | ❌ | D2（合成随机游走数据） |

---

## 下一步：唯一主要动作

**全量重跑 → 产出可进论文的新数字。**

P0 代码修复已全部完成（2026-09-03）。Colab 训练管线已修复（分支切换 + max_frames 默认值）。

### 首轮重跑结果（2026-09-04）

| 模型 | Params | Epochs | MODA | MODP | P | R | F1 | Threshold | NMS | TP/FP/FN |
|------|:------:|:------:|:----:|:----:|:---:|:---:|:---:|:---------:|:---:|----------|
| ResNet-18 + concat | 32.7M | 5 | 0.8036 | 0.7356 | **0.9682** | 0.8309 | 0.8943 | 0.225 | 8.0 | 791/26/161 |
| MobileNet-V2 + geo_cv1 | **5.7M** | 9 | **0.8445** | **0.7495** | 0.9094 | **0.9380** | **0.9235** | 0.225 | 8.0 | 893/89/59 |

**Δ (Ours − Baseline)**: MODA **+4.1pp**, MODP +1.4pp, Recall **+10.7pp**, F1 +2.9pp, Precision -5.9pp, 参数量 **5.7× 缩减**

协议：seed=42, train 0-319, val 320-359 grid-search 选超参, test 360-399 固定超参, 世界坐标 GT, greedy 0.5m

**注**：ResNet-18 仅 5 epoch（colab exec timeout），MobileNet-V2 仅 9 epoch。完整 10 epoch 待补跑。两模型 val 均选出 threshold=0.225 / NMS=8.0。

### P0 修复顺序 — ✅ 全部完成（2026-09-03）

| 顺序 | 项目 | 修复内容 | 状态 |
|:---:|------|------|:---:|
| 1 | D2 | 删除合成 fig9，重写为从 `tracker_trajectories.json` 加载真实数据 | ✅ |
| 2 | A6 | 加 `--seed` + `torch.manual_seed` + 验证集 320-359 + 每 epoch `validate()` | ✅ |
| 3 | A3 | `evaluate_detection` 从 annotation JSON 加载世界坐标 GT，匹配在米制空间 | ✅ |
| 4 | A4 | `compute_moda_modp` 从 Hungarian 改为 greedy（CLEAR MOT 标准） | ✅ |
| 5 | B1 | `colab_train.py` 分两步：验证集网格搜索 → 测试集固定超参只跑一次 | ✅ |
| 6 | C1 | `compute_velocities` 从中心差分改为后向差分 | ✅ |
| 7 | C3 | AUPRC 阈值上界从 `max(pred_max, 1e-6)` 改为 `max(pred_max, 1.0)` | ✅ |

### P1 修复（P0 全部完成后开始）

| 顺序 | 项目 | 修复内容 |
|:---:|------|------|
| 8 | B2 | 3–5 种子，报 mean ± std |
| 9 | B3 | fusion↔head 解耦 + 2×2 消融网格 + MobileNet+concat 对照 |
| 10 | B5 | 在 5.7M 截断模型上重测 latency/FPS |
| 11 | A1 | offset head 正确监督目标，或从论文中移除 |

### P2 修复（写作前）

| 顺序 | 项目 | 修复内容 |
|:---:|------|------|
| 12 | A8 | 提高输入分辨率或文档记录 |
| 13 | A7 | 移除无效 coverage 通道或改为 per-view coverage |
| 14 | B6 | 统一 README / config / 执行脚本 |
| 15 | D4 | MultiviewX 数据集 |
| 16 | D5–D9 | 基线更新、诊断证据、文档补齐、artifact 冻结 |

### P3 修复（可并行，不阻塞主流程）

| 项目 | 修复内容 |
|------|------|
| A2 | batch>1 索引修复 |
| A5 | focal 下 train/eval 一致 + 阈值区间自适应 |
| A9 | hflip 警告或拒绝 |
| A10 | 测试断言改为 `allclose` |
| A11 | numpy 版本约束 |
| A12 | 死代码清理 |

---

## 预期验证

P0 全部修复后，应产出：

```text
一张新的主表：
  - 固定超参（threshold=0.400, NMS=6.0，在验证集上选定）
  - 世界坐标 GT（非池化 GT）
  - 统一贪心匹配 + 0.5 m
  - 3 种子 mean ± std
  - 5.7M 模型上的延迟/FPS
  - 列：MVDet 复现（ResNet-18+concat）vs Ours（MobileNet-V2+geo_cv1）
```

Module 2 修复后，应产出：

```text
一张新的轨迹预测/场预测表：
  - 常速度（后向差分）vs MLP vs 其他
  - 统一阈值网格的 AUPRC
  - by-trajectory 的 std
  - non-overlapping 窗口
```

---

## Do Not Do Next

- ❌ 不要开始写论文（当前数字全部无效，写了也是废稿）
- ❌ 不要做新的模型改进（修复协议前任何新实验都是浪费）
- ❌ 不要修改 `research_progress.md` 的已有数字（保留历史记录，新数字入新迭代）
- ❌ 不要引入 BEVFormer/PETR/LSS/DETR3D 等大框架
- ❌ 不要加 tracking/ReID/轨迹预测的新模块
- ❌ 不要删除任何 `ai_runs/` 下的历史实验记录
- ❌ 不要在没有用户评审的情况下提交代码变更（AGENTS.md §16 红线）