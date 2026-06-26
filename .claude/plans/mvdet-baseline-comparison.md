# 计划：MVDet 基线对照 + 逐层对比诊断

## Context

项目经过 2.5 个月、185 个 commit、56 次 Colab run，MODA 最高仅 -0.044（目标 ≥ 0.30，MVDet 论文 88.2%）。
SNR 卡在 0.56，模型在 BEV 空间无法有效区分行人和背景。

继续调参无法弥合这个数量级差距。根因必须通过与 MVDet 官方实现的直接对比来定位——
我们从来没有在同一数据上跑过 MVDet 官方代码获取可信参考基线。

---

## Phase 1: 在 Colab 上运行 MVDet 官方代码

**目标**: 用 MVDet 官方仓库 + 我们的 WildTrack 数据，训练 10 epoch，确认能复现 ~88% MODA。

### 步骤

1. **创建 `scripts/run_mvdet_baseline.py`**
   - Clone `https://github.com/hou-yz/MVDet` 到 `/content/MVDet`
   - 安装依赖：`kornia`, `scipy`, `opencv-python`（MVDet 无 requirements.txt）
   - 软链接或拷贝 WildTrack 数据到 MVDet 期望的路径 `~/Data/Wildtrack`
   - 运行 `python main.py -d wildtrack --epochs 10 --lr 0.1`
   - 记录训练 loss、最终 MODA/MODP

2. **创建 `.github/workflows/mvdet-baseline.yml`**
   - workflow_dispatch 触发
   - 使用相同的 Colab 基础设施（colab-cli）
   - 下载 WildTrack 数据（复用现有逻辑）
   - 运行 `scripts/run_mvdet_baseline.py`
   - 将结果打印到 GA 日志

3. **预期结果**：MVDet 在 10 epoch 后 MODA ≈ 85-88%
   - 如果能复现 → 进入 Phase 2
   - 如果不能复现 → 先排查数据/环境问题

---

## Phase 2: 提取中间输出进行逐层对比

**目标**: 对同一帧图像，对比 MVDet 和我们的模型在每一层的输出，找到第一个 diverge 的位置。

### 对比节点（从前到后）

| # | 对比内容 | MVDet 提取方式 | 我们的提取方式 |
|---|---------|---------------|--------------|
| A | 投影矩阵 | `model.proj_mats` (7×3×3) | `proj_mats` tensor |
| B | Backbone 输出 | 第一个 view 的特征图 shape + stats | 同上 |
| C | 单视角 BEV 投影 | `kornia.warp_perspective` 后的特征图 | `warp_perspective_torch` 后的特征图 |
| D | 多视角 concat 后的特征图 | `world_features` concat 后 | concat 后 |
| E | BEV head 输出 (raw logits) | `map_result` | `map_logits` |
| F | GT 热图 | `gp_gt` | `map_gt` (pooled) |

### 实现方式

创建 `scripts/compare_layers.py`：
- 加载同一帧 (frame 0) 的同一组图像
- 分别在 MVDet 和我们的模型上 forward
- 对每个节点输出 shape、min/max/mean/std
- 对投影矩阵做逐元素对比 (allclose)
- 保存关键 tensor 为 .npy 文件供可视化

**关键对比**:
1. **投影矩阵**: `torch.allclose(our_proj, mvdet_proj, atol=1e-4)` — 如果这里不同，后面全白做
2. **单视角 BEV 特征**: 对比 view 0 的 BEV 特征图，可视化为热图
3. **GT 热图**: 确认我们的 GT 与 MVDet 的 GT 一致

---

## Phase 3: 修复 divergence

基于 Phase 2 的对比结果：

- **如果投影矩阵不同** → 修复 `src/geometry.py` 中的 `build_mvdet_proj_mat()`，
  重点检查 unit_scale 处理、img_zoom/map_zoom 的等效性、permutation 矩阵
- **如果投影正确但 BEV 特征不同** → 检查 `warp_perspective_torch` vs `kornia.warp_perspective` 
  的坐标约定（pixel center、归一化范围、padding mode）
- **如果特征正确但 GT 不同** → 检查 `positionID → grid` 的映射和 Gaussian 核构造
- **如果一切正确但训练结果不同** → 检查 optimizer/scheduler 的实际行为

---

## Phase 4: 验证修复

修复后的验证标准：

1. `scripts/compare_layers.py` 的所有对比节点 allclose = True
2. 在 Colab 上用修复后的代码训练 10 epoch
3. MODA > 0.30（最低可用标准）
4. 如果仍低于 MVDet baseline → 继续按层对比迭代

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/run_mvdet_baseline.py` | 新建 | Colab 上运行 MVDet 官方代码 |
| `.github/workflows/mvdet-baseline.yml` | 新建 | 触发 MVDet baseline run |
| `scripts/compare_layers.py` | 新建 | 逐层对比脚本 |
| `src/geometry.py` | 可能修改 | 如果投影矩阵有 divergence |
| `src/models.py` | 可能修改 | 如果 warp 或 fusion 有 divergence |
| `src/dataset.py` | 可能修改 | 如果 GT 构造有 divergence |

---

## 验证方法

1. `python scripts/run_mvdet_baseline.py` 在 Colab 产出 MODA ≈ 88%
2. `python scripts/compare_layers.py` 在每个节点打印 allclose 结果
3. 修复后跑训练，eval 产出 MODA > 0.30

---

## 风险

- MVDet 依赖 `kornia`，版本兼容性可能有问题（PyTorch 版本）
- MVDet 的 MATLAB eval 在 Colab 上可能跑不了，需用其 Python eval 替代
- WildTrack 数据路径需要适配 MVDet 的目录结构
- 如果 MVDet 官方代码本身跑不出 88% MODA（比如数据版本不同），需要降低预期
