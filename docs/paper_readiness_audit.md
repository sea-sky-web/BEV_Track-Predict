# Paper Readiness Audit — 论文可投稿性审计

> 审计日期：2026-09-02
> 审计范围：全仓库（`src/`、`src/temporal/`、`scripts/`、`docs/`、`tests/`、`ai_runs/`、workflows）
> 审计方法：逐文件阅读 + 测试套件实跑验证（137 测试，1 失败为浮点舍入）
> 关联文档：`docs/second_stage_innovation_review.md`（自我红队，作为本次审计的基线）
>
> **本审计不修改任何代码，仅记录发现。修复优先级和顺序见 `docs/active_plan.md`。**

---

## A. 代码层缺陷（Implementation Bugs）

### A1. offset head 回归目标恒为 0（子像素细化为空操作）

- **文件**：`src/trainer.py:221`
- **代码**：`gt_offset = torch.zeros_like(offset_preds)`
- **影响**：`offset_head` 被训练为恒输出 0，只起 L1 正则作用，不含任何亚格点位置信息。评估时 `--use_offset` 加上的是一个被训练为 0 的量。Colab 默认 `--offset_weight 0.0`（`scripts/colab_train.py:60`），产出 `research_progress.md` 头条数字的那次运行并未启用此模块。
- **正确目标**：`(Xw_gt - cell_center) / cell_size`，`dataset.py` 中已有 `Xw = ox + (ix+0.5)*step` 可计算。
- **最小修复**：在 `dataset.py` 中生成 `offset_gt`（2 通道，0.025 m 原网格），pool 到 reduced grid 做监督。
- **验证**：训练后 `offset_head` 输出均值不应为 0；`evaluate_main.py` 中 offset 开启后 MODP 应提升。

### A2. batch_size > 1 直接崩溃

- **文件**：`src/trainer.py:222`
- **代码**：`offset_preds[:, :, pos_mask_offset.squeeze(0) if B == 1 else pos_mask_offset]`
- **影响**：`offset_preds` 是 4-D `(B, 2, H, W)`，`pos_mask_offset` 是 3-D `(B, H, W)` 布尔掩码。`B == 1` 时 `.squeeze(0)` 侥幸通过；`B > 1` 时用 3-D 掩码索引 4-D 张量的 dim=2 会抛异常。整个 Module 1 事实上只能 `B=1` 训练。
- **最小修复**：改为 `offset_preds[pos_mask_offset.unsqueeze(1).expand(-1, 2, -1, -1)]`。
- **验证**：`B=2` 的 smoke test 通过。

### A3. 评估 GT 来自池化热力图，非标注

- **文件**：`src/evaluate_main.py:392-399`（`_extract_gt_points`），`src/evaluate_main.py:294`（`adaptive_max_pool2d`）
- **代码**：`pooled_gt = F.adaptive_max_pool2d(map_gt, output_size=map_logits.shape[-2:])` → `_extract_gt_points(gt_map, ...)`
- **影响**：
  1. GT 位置被量化到 0.1 m 网格（MODA 阈值 0.5 m 的 20%）；
  2. 同一 4×4 块内两人合并为一点，`n_gt` 系统性低估。MODA = 1 − (FP+FN)/n_gt，分子分母同时被污染。
- **正确做法**：从 `annotations_positions/*.json` 直接取世界坐标 GT，在米制空间匹配。
- **最小修复**：在 `evaluate_detection` 中接受 world-coordinate GT 列表，替代 `pooled_gt` 方案。
- **验证**：重新评估后 `n_gt` 应增加（之前被 max-pool 合并的恢复为独立点），MODA 可能下降。

### A4. 一张表两套匹配器 + 两个阈值

- **文件**：`src/metrics.py:10-14`（`linear_sum_assignment`）+ 0.5 m；`src/evaluate_main.py:408-431`（`_match_points` 贪心）+ `det_dist_thr=3.0` 缩放格 = 0.3 m
- **影响**：`research_progress.md` 表里的 MODA 0.8950（Hungarian + 0.5 m）和 P 0.9301 / R 0.9223（贪心 + 0.3 m）不是同一次匹配的产物，无法互推。MVDet/CLEAR 官方 MODA 用贪心匹配，这里换成了 Hungarian，会略微高估。
- **最小修复**：统一为贪心匹配 + 0.5 m，或至少保证同一张表内所有指标用同一匹配器。
- **验证**：统一后 MODA 和用 P/R 反推的预期值一致。

### A5. focal 训练下 train/eval 损失不一致

- **文件**：`src/trainer.py:234`（永远 MSE 算 image loss）；`src/evaluate_main.py:176-198`（用 BEV criterion 算 image loss）
- **影响**：`loss_type=focal` 时训练与评估 loss 不可比。更关键的是：MSE 模式下输出是 raw logits（`GaussianMSE` 直接对 logits 做 MSE，与 MVDet 一致），检测阈值扫描区间 `-0.50 … 0.60`；focal 后输出变成 sigmoid 概率，同一套阈值区间失去意义。代码里没有任何分支处理。
- **最小修复**：在 `evaluate_model` 中根据 `loss_type` 选择 image criterion；`evaluate_detection` 的阈值区间按 `loss_type` 自适应。
- **验证**：`loss_type=focal` 时阈值区间应在 `[0.01, 0.99]`。

### A6. 无随机种子、无验证集、validate() 为死代码

- **文件**：`src/train_main.py`（全文无 `torch.manual_seed`/`np.random.seed`）；`src/trainer.py`（`validate()` 实现了但从未被调用）；`src/temporal/time_utils.py`（声明 `val: (320, 360)` 但训练用 frame 0–359）
- **影响**：Module 1 无验证集，所有超参（阈值、NMS 半径）只能在测试集上选 → 直接导致 **B1**。
- **最小修复**：
  1. 加 `torch.manual_seed(seed)` 到 `train_main.py`；
  2. 训练限定 `max_frames=320`，保留 `320-359` 为验证集；
  3. 在训练循环中调用 `validate()`。
- **验证**：3 种种子下 MODA 的 max-min 差应 < 2 pp。

### A7. GeoConfidenceFusion 的 coverage_count 通道是数学 no-op

- **文件**：`src/geometry.py:284-285`（`coverage = coverage.unsqueeze(0)` 对所有视角相同）；`src/models.py:423`（`.unsqueeze(0)` 得到 `(1, V, H, W)` 依赖 batch 广播）
- **影响**：`coverage_count` 是"该格被多少视角覆盖"，对所有视角完全相同。融合权重在视角维做 softmax，视角无关的常数完全约掉 → 三通道几何先验有 1/3 是无效的。同时 `unsqueeze(0)` 依赖 batch 广播，与 A2 的 B=1 限制耦合。
- **最小修复**：移除 `coverage_count` 通道，或改为逐视角的 coverage 指标（如 `per_view_coverage`）；`.unsqueeze(0)` 改为 `.expand(B, -1, -1, -1)`。
- **验证**：`geo_confidence_v1` 在 `bev_down=8` 或 `bev_down=16` 下应有显著增益（覆盖更稀疏时几何先验更关键）。

### A8. 输入分辨率导致特征平面名义分辨率虚高

- **文件**：`src/train_main.py`（输入 720×1280）；`src/models.py`（stride-8 backbone → 90×160，bilinear upsample → 270×480）
- **影响**：270×480 的特征平面只有 90×160 的真实信息量（3× 虚假上采样）。MVDet 原文是 2×。这是与 MVDet 复现值（0.8456）差原文（0.882）3.6 个点的最可能来源。
- **最小修复**：要么提高输入分辨率（如 1080×1920 → stride-8 = 135×240，仅需 1.125× upsample 到 270×480），要么接受当前分辨率并文档记录。
- **验证**：Input 1080p 时 MODA 应接近 MVDet 原文 0.882。

### A9. hflip 增强在几何上非法

- **文件**：`src/augmentation.py:93-96`
- **代码**：`aug_imgs = torch.flip(aug_imgs, dims=(-1,)); map_gt = torch.flip(map_gt, dims=(-1,)); aux_gt = torch.flip(aux_gt, dims=(-1,))`
- **影响**：每个相机的 homography 是固定的标定量，翻转图像后投影矩阵不再对应。默认 `hflip_prob=0.0` 当前无害，但 README 写着 augmentation 开启，是随时会被打开的地雷。
- **最小修复**：在 `ViewCoherentAugment.__call__` 中当 `hflip_prob > 0` 时 emit warning 或直接 raise。
- **验证**：`hflip_prob=0.5` 时程序应警告或拒绝。

### A10. 测试套件：1 个真实失败（已实跑确认）

- **实跑结果**：`1 failed, 136 passed in 7.77s`
- **失败测试**：`tests/test_augmentation.py::test_view_coherent_hflip_flips_images_bev_and_aux_labels`
- **根因**：`color_jitter=(0,0,0,0)` 时 `_photometric` 仍走 `(x-mean)*1.0+mean`，引入 ~1.9e-9 浮点舍入，测试用 `torch.equal` 逐位比较。`torch.allclose(atol=1e-6)` 通过，`map_gt`/`aux_gt` 翻转完全正确。
- **修复**：断言改为 `torch.allclose(..., atol=1e-6)`，或在因子为恒等时短路跳过运算。
- **注意**：`.remember/now.md` 记"121/122 通过"已过期（当前 136/137）。

### A11. 依赖声明与代码不符

- **文件**：`requirements.txt:1`（`numpy>=1.23`）；`src/temporal/field_metrics.py:81`（`np.trapezoid` 需要 numpy ≥ 2.0）
- **影响**：NumPy 1.x 环境下 Module 2 的 AUPRC 计算会 `AttributeError`。
- **最小修复**：`requirements.txt` 改为 `numpy>=2.0`，或 `field_metrics.py` 中 fallback 到 `np.trapz`（numpy 1.x 兼容名）。
- **验证**：`pip install numpy==1.26.4 && python -c "from src.temporal.field_metrics import compute_occupancy_auprc"` 不报错。

### A12. 死代码与重复定义

| 问题 | 位置 | 修复 |
|------|------|------|
| `create_grid_sampler` 从未被 import | `src/geometry.py:304-345` | 标注 `# dead code` 或删除 |
| `scripts/calibration.py` 与 `src/calibration.py` 逐字节相同 | `diff` 为空 | 删除 `scripts/` 副本，全部 import 自 `src/` |
| `FRAME_RATE_HZ` 定义两次 | `src/temporal/time_utils.py` + `src/temporal/annotation_reader.py` | 统一 import 自 `time_utils` |
| `make_temporal_windows` docstring 写"non-overlapping"但 stride=1 | `src/temporal/time_utils.py` | 修正 docstring 或改为真实 non-overlapping |

---

## B. 实验协议与结论有效性

### B1. 头条 MODA 是对测试集 132 组超参取 argmax

- **文件**：`scripts/colab_train.py:234-235`（22 阈值 × 6 NMS 半径 = 132 配置）；`src/evaluate_main.py:505`（`best = max(rows, key=...)`）；`src/evaluate_main.py:723`（`global_best = max(by MODA)`）
- **影响**：全 132 组配置在测试集（frame 360–399）上评估后取最大值。各方法最优配置不同（0.425/NMS 6.0 vs 0.275/NMS 7.0）。仓库自己的 `docs/second_stage_innovation_review.md §P1` 已立规矩：**"headline comparison 使用固定 threshold=0.400、det_min_distance=6.0"**，`research_progress.md` 违反了自己的规则。
- **修复**：引入验证集（frame 320–359），在验证集上选超参，测试集只跑一次固定配置。
- **预估影响**：MODA 会下降，下降幅度未知。

### B2. 单次运行、40 帧测试集，方差与增益同量级

- **证据**：`docs/research_progress.md §10.7` 记录 MODA 0.8950 → 复查 0.8634，归因 cuDNN 非确定性 → **3.2 pp 抖动**。声称的增益是 **+4.9 pp**（0.8950 vs 0.8456）。
- **影响**：噪声量级和信号量级同阶，当前证据不足以支撑"我们的方法更好"。
- **修复**：至少 3–5 种子，报 mean ± std。前置条件是 A6（加种子）。

### B3. 头条对比同时动了三个变量（backbone + fusion + BEV head）

- **文件**：`src/models.py:550-553`
- **代码**：`if fusion_mode == "concat": self.bev_head = MVDetMapClassifier(...) else: self.bev_head = BEVHeadDilated(...)`
- **影响**：fusion 与 head 无法解耦（P0.5，仍开放）。关键对照组 MobileNet-V2 + concat 因 OOM 缺失（3586 通道）。无法排除"增益全部来自 BN + dilated head"。
- **修复**：
  1. 将 `bev_head` 从 fusion mode 解耦为独立参数；
  2. 跑 2×2 网格 `{concat, geo_cv1} × {MVDetMapClassifier, BEVHeadDilated}`；
  3. 用梯度检查点或降通道解决 MobileNet + concat 的 OOM。

### B4. 0.8950 vs MVDet 原文 0.882 不可比

- **差异**：
  1. 自己的 MVDet 复现只有 0.8456（−3.6 pp gap，见 A8）；
  2. 池化 GT（A3）、Hungarian MODA（A4）、测试集调参（B1）；
  3. 输入 720×1280（非原文设置）。
- **修复**：论文中唯一可用的对比是"自己复现的 0.8456 vs 自己的方法"，固定超参。原文 0.882 只能作为 Related Work 引用，不能放在同一张比较表里。

### B5. 延迟/FPS 在另一个模型上测的

- **证据**：`docs/research_progress.md §3.4` 自注 benchmark 用**未截断** MobileNet-V2（8.0M），不是产出 MODA 0.8950 的截断版（5.7M）。
- **影响**："−35% 延迟 / +55% FPS" 与精度数字不属于同一模型，不能写进同一张表。
- **修复**：在 5.7M 截断模型上重测 latency/FPS。

### B6. README / config / 实际执行脚本三方分歧

| 来源 | `max_frames` | 学习率 | 优化器 | 训练范围 | 泄漏测试集？ |
|------|:---:|:---:|:---:|---|---|
| `README.md` | −1 | 0.05 | — | 0–399 | **是** |
| `src/train_main.py` | −1 | 0.1 | SGD | 0–399 | **是** |
| `configs/exp_colab.yaml` | −1 | — | Adam | 0–399 | **是** |
| `scripts/colab_train.py` | 360 | 0.1 | SGD | 0–359 | **否** |

- **影响**：产出数字的那次运行是干净的（`colab_train.py`），但按 README/YAML 复现的人会训在测试集上，得到一个更高的、无意义的 MODA。
- **修复**：`src/train_main.py` 默认 `max_frames=320`（留 320–359 为验证集），`exp_colab.yaml` 同步更新，README 指向 `colab_train.py`。

---

## C. Module 2（时序/预测）问题

### C1. 常速度基线泄漏未来（"MLP 输给常速度"无效）

- **文件**：`src/temporal/annotation_reader.py:138`（`vel[i] = (pos[i+1] - pos[i-1]) / (2*dt)` 中心差分）；`src/temporal/trajectory_predictor.py:81-82`（取 `last_idx_in_traj` 处的速度）
- **影响**：被评估轨迹的最后一个已观测帧处的中心差分用到 `pos[i+1]`，即第一个预测目标。ADE=0.1555 m / FDE=0.2693 m 因此虚低。"MLP（0.3358）不如常速度"不成立。
- **修复**：末点改用后向差分 `vel[i] = (pos[i] - pos[i-1]) / dt`。
- **预估影响**：常速度 ADE/FDE 会上升，MLP 可能反而胜出。

### C2. ±std 和 N 不是独立样本

- **文件**：`src/temporal/trajectory_predictor.py:115-172`
- **影响**：`ade_std` 是跨 33 个完全重叠（stride=1）窗口的 std，同一人被数 33 次。`n_trajectories=498` 是各窗口人数求和。±0.0360 不能当误差棒，498 不能当样本量。
- **修复**：改为 non-overlapping 窗口，或报告 by-trajectory 的 std（每条轨迹只计一次）。

### C3. AUPRC 实现依赖预测值量级，两个结论同时被污染

- **文件**：`src/temporal/field_metrics.py:59,81`
- **代码**：`thresholds = np.linspace(0.0, max(pred_max, 1e-6), n_thresholds+1)`；`np.trapezoid(precisions, recalls)`
- **影响**：
  - ConvLSTM（sigmoid 输出 ~[0,1]）与 Persistence/Advection（occ_max=0.045）阈值网格粒度差 20× 以上；
  - "ConvLSTM 比基线差 17×/25×"（B7）无法区分是模型真差还是 AUPRC 尺度偏差；
  - "bev_down=16 救回 ConvLSTM"（B9）同样被混淆（改分辨率也改了 field 量级）。
- **修复**：统一阈值网格（如 `np.linspace(0, 1, 100)`）或改用 `sklearn.metrics.average_precision_score`。重跑 B7 和 B9 两组实验。
- **预估影响**：ConvLSTM 原分辨率 AUPRC 可能从 0.0301 大幅上升。

---

## D. 论文未完成项（Non-code Gaps）

### D1. 无论文草稿

全仓库搜索 `*.tex` / `*paper*` / `*draft*` 只匹配到 `scripts/generate_paper_figures.py`。目前有实验记录（`research_progress.md`）+ 自我红队（`second_stage_innovation_review.md`），没有手稿。摘要、Related Work、方法形式化、实验分析、结论——一个字都还没写。

### D2. fig9_tracking.png 是合成数据（学术诚信问题）

- **文件**：`scripts/generate_paper_figures.py:435-483`
- **代码**：
  ```python
  def fig_tracking_vis():
      """Fig 9: Simulated BEV tracking visualization with realistic curved trajectories."""
      rng = np.random.default_rng(42)
      ...
      ax.text(..., "Kalman+Hungarian Tracker\n15 active tracks, IDSW=0", ...)
  ```
- **影响**：随机游走的数据，配了一段真实追踪器结果的图注。发表后是学术不端。
- **修复**：从真实 tracker 输出（`ai_runs/` 中的检测点 JSONL）生成此图，或删除该图。

### D3. 所有图表硬编码数值

`scripts/generate_paper_figures.py` 全文硬编码（`auprc = [0.5224, 0.7645, 0.0301, 0.663]`、`t_ade = [0.1555, 0.3358]`…），而非从 `ai_runs/*/metrics.json` 读取。一旦 A/B/C 组修复后重跑，所有图与正文数字脱节。
- **修复**：改为从 `ai_runs/` 读取最新 `metrics.json`。

### D4. 只有一个数据集

WildTrack 单数据集 + 40 帧测试集。MultiviewX 是标准第二数据集，投影代码复用度高。
- **修复**：添加 MultiviewX 支持。

### D5. 基线太旧

只对比了 MVDet（ECCV 2020）。2021 年后有 MVDeTr、SHOT、EarlyBird 等，WildTrack MODA 已推到 0.91+。本工作的真正卖点是"轻量化"（5.7M vs 32.7M）而非精度 SOTA，但需要 B5 修好、参数量/FLOPs/延迟三项对齐后才能主打。
- **修复**：Related Work 中如实报告这些方法的公开数字，强调本文贡献是轻量级设计。

### D6. 缺乏几何先验的诊断证据

`docs/second_stage_innovation_review.md §P1` 自己列了：按 `coverage_count` 分层、按 `border_margin` 分箱的 MODA 分解。没有它，`geo_confidence_v1` 相对 `confidence_v2` 的 +0.3 点（0.8950 vs 0.8918，远小于 B2 的 3.2 pp 噪声）完全没有说服力。
- **修复**：A7 的无效通道先修，然后跑分层诊断。

### D7. 文档链条断裂

`docs/model_definition.md` 引用了 `docs/experiment_protocol.md`、`docs/evaluation_protocol.md`、`docs/research-methodology.md`，这三个文件都不存在。
- **修复**：创建这三个文件。

### D8. 没有冻结的 artifact

没有 tag / release / 固定 commit 标识"论文里的 baseline 是哪一版"。加上 A6 的种子缺失，无法回答"表 1 是哪次运行产生的"。
- **修复**：固定 commit → tag → 归档 checkpoint SHA256 到 `docs/m1_frozen_detector_manifest.md`。

### D9. P0.1 和 P0.5 仍开放

`docs/second_stage_innovation_review.md` 已记录的两个问题，代码核验后确认未解决：
- **P0.1**：`src/models.py:503` `mid_ch=128`，文档写 `mid_ch=64`
- **P0.5**：`src/models.py:550-553` fusion↔head 绑定

---

## 修复优先级

| 优先级 | 项目 | 理由 |
|:---:|------|------|
| 🔴 P0 | B1（测试集调参） | 不修则所有结果作废，审稿人当场拒 |
| 🔴 P0 | A3（池化 GT） | 评估协议的根本缺陷 |
| 🔴 P0 | D2（fig9 合成数据） | 学术诚信问题 |
| 🔴 P0 | C1（常速度泄未来） | Module 2 核心结论可能反转 |
| 🔴 P0 | C3（AUPRC 尺度偏倚） | Module 2 另外两个核心结论也可能反转 |
| 🟠 P1 | A6（种子+验证集） | B2 的方差问题必须解决 |
| 🟠 P1 | B3（fusion↔head 解耦） | 核心主张没有对照证据 |
| 🟠 P1 | A4（统一匹配器） | 一张表不能有两套逻辑 |
| 🟠 P1 | B5（延迟对齐） | 精度和效率必须同一模型 |
| 🟠 P1 | A1（offset head） | 要么修复，要么从论文中移除 |
| 🟡 P2 | A8（输入分辨率） | 解释 MVDet 复现 gap |
| 🟡 P2 | A7（无效 coverage 通道） | 几何先验的诊断证据前提 |
| 🟡 P2 | B6（README 分歧） | 可复现性 |
| 🟡 P2 | D4–D9 | 论文完整性和竞争力 |
| 🟢 P3 | A2, A5, A9, A10, A11, A12 | 不影响当前结论但应修复 |

---

## 一句话总结

**当前 `research_progress.md` 表格里没有一个数字可以直接进论文。** B1（测试集调参）、A3（池化 GT）、A4（双匹配器）各自独立地就足以让主表作废；B2（方差与增益同量级）进一步要求全量重跑。距离可投稿状态，保守估计需要 P0 修复 + 全量重跑 → P1 消融补全 → 写作 → P2 竞争力提升。`fig9_tracking.png` 请务必优先处理，那个不是技术问题。