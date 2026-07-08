# Plan: Geometry-Reliability Prompted Fusion (`geo_confidence_v1`)

## Context

当前 BEV 行人检测 baseline MODA=0.849-0.857，目标超越 MVDet 论文的 0.882。`warp_perspective_torch` 在将图像特征投影到 BEV 时，每个视角对每个 BEV 位置的可靠性不同（视野外、图像边缘、畸变区域），但这些几何信息在投影后被丢弃。本方案将静态几何可靠性先验注入融合路径，引导模型降低不可靠 view/cell 对的权重。

## 实现方案

### 1. `src/geometry.py` — 添加 `compute_bev_geometry_metadata`

在 `compute_valid_ratio_from_homography` 后添加。纯函数，无梯度，只在模型初始化时调用一次。

```python
def compute_bev_geometry_metadata(
    proj_mats: torch.Tensor,     # (V, 3, 3)
    src_hw: Tuple[int, int],     # (Hf, Wf)
    dst_hw: Tuple[int, int],     # (Hb, Wb)
) -> torch.Tensor:               # (V, 3, Hb, Wb)
```

复用 `warp_perspective_torch` 中相同的逆投影数学（inv(M) @ dst_grid），对每个 view 每个 BEV cell 计算：
- **ch0 valid_mask** (float 0/1)：源坐标有限、z>1e-6、在图像范围内
- **ch1 border_margin** (float 0-1)：`min(x, Ws-1-x, y, Hs-1-y) / (min(Hs,Ws)/2)`，无效处=0
- **ch2 coverage_count** (float 0-1)：所有视角 valid_mask 之和 / V，每个 view 相同

返回 `(V, 3, Hb, Wb)` float32。

### 2. `src/models.py` — 添加 `GeoConfidenceFusion` 类

在 `ConcatAttentionFusion` 类后添加。继承 `confidence_v2` 的 joint feature scoring 路径，额外增加几何评分：

```
feature_scores = feature_weight_head(joint_compress(cat(feats)))  # (B,V,H,W)
geo_scores = geo_score_net(geo_meta_per_view)                    # (1,V,H,W)
combined = feature_scores + beta * geo_scores
weights = softmax(combined, dim=view)
output = sum_v(weights * feats)
```

- `geo_score_net`: Conv2d(3, 1, 1) — 仅 4 个参数
- `beta`: nn.Parameter(1.0) — 可学习

### 3. `src/models.py` — 修改 `MVDetLikeNet`

- `FusionMode` 类型添加 `"geo_confidence_v1"`
- `normalize_fusion_mode` 不变（新模式不需要别名）
- `__init__` 中：
  - 新增 `elif fusion_mode == "geo_confidence_v1"` 分支，创建 `GeoConfidenceFusion`
  - 调用 `compute_bev_geometry_metadata`，结果 `register_buffer("geo_meta")`
  - 使用 `BEVHeadDilated`（与 confidence_v2 一致，公平对比）
- `forward` 中：
  - 非 concat 分支增加判断：`geo_meta is not None` 时传入 fusion 调用
- `__init__` 中的 fusion_mode 校验集合添加 `"geo_confidence_v1"`

### 4. CLI — `train_main.py` 和 `evaluate_main.py`

两个文件各改一处：`--fusion_mode` 的 `choices` 列表添加 `"geo_confidence_v1"`。

### 5. 测试

**`tests/test_geometry.py`**：
- 添加 `test_compute_bev_geometry_metadata_shapes_and_values`（identity proj → 全部 valid, margin>0, coverage=1.0）

**`tests/test_smoke_forward.py`**：
- 在 parametrize 列表添加 `"geo_confidence_v1"`（自动覆盖 forward shape 检查）
- 添加 `test_geo_confidence_beta_has_gradient`

## 不做的事

- 不修改 `warp_perspective_torch`
- 不修改任何现有 fusion 模块
- 不添加 Jacobian/projective-scale（先验证 valid_mask+margin 是否有效，后续按需加 ch）
- 不改 colab_train.py（训练时再改 fusion_mode 参数）

## 验证

```bash
# 本地测试
PYTHONPATH=src python3 -m pytest tests/test_geometry.py tests/test_smoke_forward.py tests/test_loss.py -v

# CPU smoke（如有 wildtrack 数据）
PYTHONPATH=src python3 src/train_main.py --data_root wildtrack --views 0,1 --device cpu --max_frames 2 --fusion_mode geo_confidence_v1 --epochs 1
```
