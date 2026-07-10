# 经验沉淀 — LESSONS.md

> Append-only。执行前必读，避免重蹈覆辙。
> 分两部分：方法论规则 + 具体实验记录。

---

# Part A: 方法论规则

> 从 2026-05-04 到 2026-07-07 的排查过程中总结

---

## 1. 实证优先，禁止"可能"

**教训来源**：07-06 排查中，将 workflow 默认值问题标记为"可能是根因"，用户要求实证验证。通过查看上次 run 的实际 `sys.argv` 日志，证明 workflow 默认值并非根因。

**规则**：
- 每一个假设都必须有可验证的证据链
- "可能"、"大概率"等表述不可接受，必须用实际数据确认或否定
- 验证方法：查 GA 日志中的实际命令行参数、查代码中的实际默认值、查运行结果中的实际数值

---

## 2. 坐标系一致性 — 最隐蔽的 bug 类型

**教训来源**：
- 06-29：BEV H/W 转置（NB_WIDTH=480 vs 正确值 1440）
- 07-06：NMS 半径在 reduced grid（10cm/格）上用了 full grid（2.5cm/格）的数值

**规则**：
- 涉及坐标系的代码，必须标注每个变量所在的坐标系和物理单位
- 不同坐标系之间的转换（full grid ↔ reduced grid ↔ 世界坐标米/厘米）必须显式进行
- 复制参数值时，不能只复制数值，必须同时确认数值的坐标系语义

**检查清单**：
- [ ] 这个数值是在哪个坐标系中的？
- [ ] 物理单位是什么？（格数、米、厘米）
- [ ] 目标代码使用的坐标系与来源一致吗？

---

## 3. 逐层对比脚本的盲区

**教训来源**：`compare_layers.py` 报告 ALL PASS，但实际仍有 MODA 0.44 vs 0.88 的差距。

**遗漏的维度**：
- ✅ 对比了：投影矩阵、coord map、GT 热图、Gaussian 核
- ❌ 没对比：NMS 后处理参数、检测提取流程、评估协议（距离阈值、坐标系）
- ❌ 没对比：warp 实际输出（只比了投影矩阵，没比 warp 后的特征图）
- ❌ 没对比：loss 函数在相同输入下的输出值

**规则**：
- 对比脚本要覆盖**完整 pipeline**，从数据加载到最终指标
- 特别注意**后处理和评估代码** — 这些往往被认为"不重要"但实际影响巨大
- 对比不能只看"形状一致+统计量相近"，要比较实际数值

---

## 4. 错误的排查方向也是有价值的信息

**教训来源**：07-06 先排查 AMP、color jitter、backbone dilation，修复后 MODA 几乎不变（0.441→0.444）。这些方向虽然错误，但证实了这些因素不是瓶颈。

**规则**：
- 实验结果为"无效果"也是重要信息，必须记录
- 排除的方向可以缩小后续搜索空间
- 不要因为方向错误就不记录 — 未来可能再次遇到类似假设

---

## 5. 查看原始实现，而非依赖二手描述

**教训来源**：NMS bug 是因为注释 `(MVDet: 50/2.5=20 cells = 2.0m)` 中的 "2.0m" 是错误的。实际应该是 0.5m（20 full grid cells × 2.5cm = 50cm）。

**规则**：
- 永远要读原始代码（`git clone` MVDet 仓库，读 `nms.py`、`trainer.py`、`CLEAR_MOD_HUN.py`）
- 不要依赖注释、文档、或之前会话的总结来推断参数含义
- 参数的语义（坐标系、单位）必须从使用该参数的代码上下文中推导

---

## 6. 从观测数据反推根因

**教训来源**：MODA=0.44 时，TP=449, FP=26, FN=503。高 Precision + 低 Recall 这个模式直接指向"过度抑制"，而非"模型太弱"。

**规则**：
- 先看 TP/FP/FN 分解，再推测原因
- 高 Precision + 低 Recall → 检测器过于保守（阈值太高或 NMS 太激进）
- 低 Precision + 高 Recall → 检测器过于激进（阈值太低或 NMS 太松）
- 两者都低 → 模型本身能力不足

---

## 7. 参数调优的正确方法

**教训来源**：NMS 半径从 5.0 调到 6.0，MODA 从 0.808 提升到 0.857。

**规则**：
- 一次只调一个维度，保持其他固定
- 使用网格扫描而非手动试错
- 记录每个组合的完整指标，不只看最终最优值
- 最优参数可能不在直觉预期的位置（6.0 比"精确匹配 MVDet"的 5.0 更优）

---

# Part B: 已尝试实验及结论

> 格式：试了 X → 结果 Y → 结论 Z
> 新条目追加在最后

---

## B1. 移除 AMP 混合精度 (07-06)

- **假设**：AMP float16 精度不足 + OneCycleLR scheduler 失步导致 MODA 低
- **实验**：移除 `--amp` 标志，其他配置不变
- **结果**：MODA 0.441 → 0.444（几乎无变化）
- **结论**：AMP 不是瓶颈。float32 训练在当前配置下与 AMP 效果一致
- **验证方法**：对比 run 28560562597（有 AMP）vs run 28772291574（无 AMP）的实际 sys.argv 和 MODA

## B2. 禁用 Color Jitter 增强 (07-06)

- **假设**：MVDet 无增强，color jitter 在 10 epoch 内引入噪声阻碍收敛
- **实验**：`--augment false`，其他配置不变
- **结果**：MODA 0.441 → 0.444（与 B1 同一 run，无法单独归因，但整体无效果）
- **结论**：color jitter 不是当前瓶颈。可在后续创新阶段重新启用

## B3. 移除 Backbone Dilation（全部置为 1）(07-06)

- **假设**：MVDet 旧版 torchvision BasicBlock 忽略 dilation
- **实验**：`_undilate_basic_resnet_layer` 只保留 stride=1，不设 dilation
- **结果**：MODA 0.441 → 0.444（同 run，无显著效果）
- **结论**：后续发现 MVDet **确实使用渐进式 dilation**（从源码 resnet.py 确认），此修复方向错误。已在 PR #77 中改为精确匹配 MVDet 的渐进模式
- **教训**：必须读原始代码，不能推测"旧版忽略 dilation"

## B4. NMS 半径修正 20→5 (07-06) ★ 关键发现

- **假设**：`det_min_distance=20` 在 reduced grid 上 = 2.0m，是 MVDet 0.5m 的 4 倍
- **实验**：`det_min_distance=5.0`
- **结果**：MODA 0.441 → **0.793**（+0.35！）Recall 0.456 → 0.900
- **结论**：NMS 过度抑制是 MODA 0.44 的根因。2.0m 半径杀掉了拥挤区域的大量有效检测
- **验证方法**：MVDet 源码 `nms.py:dist_thres=50/2.5=20` 是 full grid cells，我们错误地用在 reduced grid

## B5. NMS 半径 + 阈值网格扫描 (07-07)

- **实验**：6 个 NMS 半径 × 22 个阈值 = 132 组合
- **结果**：最优 NMS=6.0, threshold=0.400, MODA=**0.857**
- **结论**：NMS=6.0（0.6m）比精确匹配 MVDet 的 5.0（0.5m）更优。FP 从 95 降到 53
- **NMS 半径影响**：3.0→过多 FP；5.0→均衡；6.0→最优；8.0→过多 FN

## B6. Backbone 渐进式 Dilation 对齐 (07-07, pending)

- **假设**：MVDet 自定义 BasicBlock 对 conv1 使用渐进 dilation (L3.B1=2, L4.B0=2, L4.B1=4)
- **实验**：PR #77, run 28845973141（进行中）
- **结果**：待确认
- **预期**：MODA 接近 0.882 则 pipeline 验证通过

---

## 8. ExitPlanMode ≠ 评审通过

**教训来源**：07-07 实现 Focal Loss 时，ExitPlanMode 批准后直接开始写代码，跳过了 `docs/rules/review-protocol.md` 要求的用户方案评审环节。

**规则**：
- `ExitPlanMode` 只表示"计划文件写完了"，不是用户评审通过
- 计划批准后、写代码前，必须：(1) 展示方案核心变更点 (2) 等用户评审意见 (3) 确认用户理解变更影响
- 这两个门禁不可合并、不可跳过

**检查清单**：
- [ ] ExitPlanMode 已通过？→ 进入评审，不是进入实现
- [ ] 用户给出实质性评审意见？→ 才可开始写代码
- [ ] 用户只说"可以/同意"？→ 追问一个验证问题确认理解

---

## 9. workflow_dispatch 的 `--ref` 只决定 checkout，不决定 Colab 侧代码

**教训来源**：07-09 Run B（focal loss，`--ref fix/focal-loss-pos-mask`）复现了未修复前的爆炸行为，尽管 GA runner 已 checkout 到含修复的分支。

**根因**：`scripts/colab_train.py` 在 Colab session 内部执行 `git clone https://github.com/.../BEV_Track-Predict.git`，**只 clone 默认分支（main）**，不会 checkout 到触发 workflow 时指定的 `--ref`。GA runner 上的代码状态和 Colab 上实际执行的代码状态是两个独立的 git 副本。

**规则**：
- 任何要在 Colab 上验证的修复，必须先合并到 `main`，再触发 workflow
- 不要假设 `gh workflow run --ref <branch>` 能让 Colab 侧跑到该分支的代码
- 触发前检查：`git log --oneline main -1` 确认修复已在 main HEAD

**检查清单**：
- [ ] 修复代码是否已合并到 main？
- [ ] 如果只是分支验证，会不会在 Colab 侧被 `git clone` 默认分支覆盖？

---

## 10. 归一化方式不同的 loss 换算前必须做梯度量级实证

**教训来源**：07-09/07-10 Run B / Run BB，focal loss（`PenaltyReducedFocalLoss`）从 epoch 0 起完全爆炸（loss 卡在 13.82，raw_pos_mse ~10¹⁹-10²⁰），即使 pos_mask 的 `eq(1.0)` bug（PR #84）已修复。

**根因**：
1. `pos_mask = tgt.eq(1.0)` 在 `F.conv2d` 模糊后几乎永远不命中（浮点精度）+ 邻近行人高斯核叠加可超过 1.0 → PR #84 已修复（`clamp(max=1.0)` + `ge(1.0-1e-4)`）
2. **更根本的问题**：focal loss 用 `loss.sum() / num_pos`（num_pos ~20-40 像素）归一化，MSE 用 `mean()`（÷43200 像素）归一化。本地实证测得两者梯度相差 **300-3600 倍**。CenterNet 原始配置搭配 Adam lr=1.25e-4，我们用 SGD lr=0.1（差 ~800 倍）

**规则**：
- 引入任何新 loss 函数前，**必须**先用相同随机种子/相同输入本地跑一次前向+反向，比较 `loss.item()`、`grad.norm()`、`grad.abs().max()` 与现有 baseline loss 的量级差异
- 如果差异超过 10 倍，lr 和 optimizer 大概率需要独立调整，不能直接复用现有训练配置
- CenterNet 类 loss 论文中的超参（lr, optimizer）是和其归一化方式配套设计的，不能只抄 loss 公式，丢弃配套的优化器设置

**验证方法**：
```python
pred = torch.randn(...) * 0.01  # 模拟接近 0 的初始 logits
loss_a = criterion_a(pred.clone().requires_grad_(), target, kernel); loss_a.backward()
loss_b = criterion_b(pred.clone().requires_grad_(), target, kernel); loss_b.backward()
# 比较 grad.norm(), grad.abs().max()
```

---

## B7. Focal Loss pos_mask eq(1.0) bug 修复 (07-09, PR #84)

- **假设**：`pos_mask = tgt.eq(1.0)` 在高斯模糊后的 GT 上匹配 0 个像素，导致 `num_pos` clamp 到 1，梯度爆炸
- **实验**：本地构造模拟 GT，验证 `eq(1.0)` 匹配数 = 0（含近邻行人叠加导致 max=1.905 > 1.0）
- **修复**：`tgt.clamp(max=1.0)` + `pos_mask = tgt.ge(1.0 - 1e-4)`
- **结果**：修复后每个行人产生 ~11 个正样本像素（本地验证），17/17 单元测试通过
- **Colab 验证**：Run BB (29083224880) 仍然爆炸 → 证明这不是唯一根因，见 B8

## B8. Focal Loss 梯度量级 300-3600x 于 MSE (07-10, PR #85 待批准)

- **假设**：即使 pos_mask 修复，`sum()/num_pos` 归一化产生的梯度量级和 SGD lr=0.1 不匹配
- **实验**：本地相同随机种子对比 MSE vs Focal 前向+反向：loss 比 49144x，grad_norm 比 3674x，grad_max 比 334x
- **结论**：CenterNet 用 Adam lr=1.25e-4 搭配此归一化；我们用 SGD lr=0.1，相差 ~800x，与实测梯度比量级吻合
- **修复方案**：PR #85 给 workflow 加 `lr_init` 参数入口（默认 0.1 不影响 MSE baseline），后续用 `lr_init=0.001` 重跑验证
- **待验证**：Colab 训练 run（需用户批准后触发）
