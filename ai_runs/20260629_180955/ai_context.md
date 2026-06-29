# AI Context — 逐层对比诊断结果

## 实验信息

| 字段 | 值 |
|------|-----|
| GA Run ID | 28354087066 (compare), 28359760126 (compare fix) |
| 日期 | 2026-06-29 |
| Git commit | 18f4eff |
| 方法 | scripts/compare_layers.py 逐层对比 MVDet 官方 vs 我们的实现 |

## 对比结果

| 节点 | 结果 | 详情 |
|------|------|------|
| A. 投影矩阵 (7×3×3) | **PASS** ✅ | 所有 7 个视角 max_diff < 0.0004 |
| B. Coord map | **FAIL** ❌ | ours=(1,2,360,120) vs MVDet=(1,2,120,360) — H/W 转置 |
| C. GT 热图 | **FAIL** ❌ | ours=57 非零点 vs MVDet=38，max_diff=1.0 |
| D. Gaussian kernel | **FAIL** ❌ | ours mean=0.093 vs MVDet mean=0.019 — kernel 宽约 5 倍 |

## Bug 1（致命）：BEV 网格 H/W 转置

### 现象
coord_map shape: ours=(1,2,**360,120**) vs MVDet=(1,2,**120,360**)

### 根因
config.py 中 NB_WIDTH 和 NB_HEIGHT 定义与 MVDet 的含义相反：

```
MVDet:  worldgrid_shape = [480, 1440]  →  H=480(N_row), W=1440(N_col)
我们:   NB_WIDTH = 480, NB_HEIGHT = 1440  →  把 480 当了 Width
```

WildTrack 使用 ij-indexing：480 是 i 方向（行/height），1440 是 j 方向（列/width）。
我们把 i 方向的 480 错误标记为 WIDTH。

### 影响
`reduced_hw = (NB_HEIGHT//4, NB_WIDTH//4) = (360, 120)` 而 MVDet 为 `(120, 360)`。
整个 BEV 空间的宽高是反的。warp_perspective 的目标网格形状错误，
所有视角的 BEV 特征在错误的空间维度上对齐。

### 受影响的文件
- src/config.py: NB_WIDTH, NB_HEIGHT 定义
- src/dataset.py: map_gt 构造用了 self.nb_h, self.nb_w
- src/train_main.py: Hb, Wb 计算
- src/evaluate_main.py: hb, wb 计算
- scripts/*.py: 所有引用 NB_HEIGHT/NB_WIDTH 的地方

### 修复
将 NB_WIDTH 改为 480→1440，NB_HEIGHT 改为 1440→480，保持变量含义：
WIDTH = 列方向 = 1440，HEIGHT = 行方向 = 480。

## Bug 2：GT 热图坐标不一致

### 现象
非零点数: ours=57 vs MVDet=38，位置完全不同

### 根因
与 Bug 1 相关。positionID 的解码：
```python
ix = pos_id % self.nb_w   # 用了 nb_w=480(错)，应该是 480(对)
iy = pos_id // self.nb_w  # 用了 nb_w=480(错)
```

MVDet: `grid_x = pos % 480, grid_y = pos // 480`

我们的 nb_w=NB_WIDTH=480 碰巧和 MVDet 的 480 一致，
但 map_gt 的形状 `(1, nb_h=1440, nb_w=480)` 是反的（MVDet 为 `(480, 1440)` reduce 后 `(120, 360)`）。

修复 Bug 1 后，map_gt 形状变为 `(1, 480, 1440)`，
positionID 解码也需要用 `ix = pos % 480, iy = pos // 480`（MVDet 一致）。
需要验证修复后非零点数为 38。

## Bug 3：Gaussian kernel sigma 解释错误

### 现象
kernel mean: ours=0.093 vs MVDet=0.019 — 我们的 kernel 宽约 5 倍

### 根因
MVDet 使用 scipy.stats.multivariate_normal.pdf：
```python
kernel = multivariate_normal.pdf(pos, [0,0], np.identity(2) * map_sigma)
# map_sigma = 20/grid_reduce = 5.0
# identity(2) * 5.0 → 协方差矩阵，variance=5.0，sigma=sqrt(5)≈2.236
```

我们的 build_gaussian_kernel_2d：
```python
g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
# sigma=5.0 直接当标准差用
```

所以我们的高斯 σ=5.0，MVDet 的 σ=√5≈2.236。我们宽了 5.0/2.236≈2.24 倍。

### 影响
GT 热图过于弥散，正样本区域扩散到远大于 MVDet 的范围。
模型学到的 peak 不够锐利，SNR 提不上去。

### 修复
将 DEFAULT_MAP_SIGMA 改为 sqrt(5.0)≈2.2361，DEFAULT_IMG_SIGMA 改为 sqrt(2.5)≈1.5811。
同时更新注释说明这是方差（variance）而非标准差。

## 这三个 bug 为什么导致 MODA 无法超过零

1. Bug 1 让 BEV 空间的几何对齐完全错误——7 个视角的特征在一个宽高反转的网格上叠加，
   空间一致性被破坏，模型无法学到有效的多视角融合信号。

2. Bug 2（部分由 Bug 1 引起）让 GT 标注和实际行人位置不对齐。

3. Bug 3 让 GT 峰过于弥散，模型难以区分行人和背景，SNR 天花板被压低。

三个 bug 叠加，模型在一个几何错乱、标注偏移、目标模糊的空间里训练了 2.5 个月。
