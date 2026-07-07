# MVDet 对齐修复计划

> **创建日期**：2026-06-25
> **状态**：进行中
> **目标**：将本项目实现与 MVDet 原始代码对齐，消除导致检测指标不达标的结构性差异
> **参考仓库**：[hou-yz/MVDet](https://github.com/hou-yz/MVDet) (ECCV 2020)
> **关联文档**：[training_goals.md](training_goals.md) · [eval_pipeline_analysis.md](eval_pipeline_analysis.md) · [model_definition.md](model_definition.md)

---

## 1. 背景

### 1.1 当前状况

Run #46（20 epoch, bev_pos_weight=10.0）的本地 eval 结果：

| 指标 | 当前值 | 目标值 | MVDet 论文 |
|------|--------|--------|-----------|
| MODA | **-0.045** | ≥ 0.30 | 0.888 |
| Precision | 0.215 | ≥ 0.50 | 0.945 |
| Recall | 0.034 | ≥ 0.40 | 0.938 |
| F1 | 0.059 | ≥ 0.40 | — |

训练过程 SNR=0.568（正样本区域 logit 均值仅比背景高 0.57），模型区分能力不足。

### 1.2 问题本质

通过逐文件对比 MVDet 原始代码（`/tmp/mvdet_ref/`），发现问题**不是超参数调优能解决的**，而是多个结构性差异的叠加：

- 融合架构根本不同
- 优化器策略完全不同
- backbone 空洞卷积实现有偏差
- 检测后处理参数不匹配

---

## 2. 关键差异详细对比

### 2.1 融合架构（最关键差异）

**MVDet 原始：concat 融合**

```python
# persp_trans_detector.py
# 所有视角特征 + 坐标编码直接 concat
world_features = torch.cat(world_features + [self.coord_map...], dim=1)
# 输入通道 = 512 * 7 + 2 = 3586
map_result = self.map_classifier(world_features)
```

BEV head 能**同时看到所有 7 个视角的全部 512 通道特征**，在每个空间位置自主学习该信任哪个视角。

**我们的实现：confidence_v2 融合**

```python
# models.py ConcatAttentionFusion
joint = feats_bev.reshape(b, v * c, h, w)
joint = self.joint_compress(joint)       # V*512 → 512
weights = softmax(self.weight_head(joint), dim=1)
fused = (feats_bev * weights).sum(dim=1)  # → 512 通道
# BEV head 输入 = 512 + 2 = 514
```

BEV head 只看到**已经被压缩到 514 通道的混合特征**，跨视角的对比信息被融合模块吞掉了。

**影响**：这是最大的架构差异。concat 模式下 BEV head 的参数量和信息容量远大于 confidence_v2，直接影响模型的空间推理能力。

### 2.2 BEV Head 结构

| 对比项 | MVDet 原始 | 我们的实现 |
|--------|-----------|-----------|
| 输入通道 | **3586**（concat 模式） | 514（confidence_v2） |
| 层数 | 3 层 conv | 4 层 conv |
| 中间通道 | **512** | 256 |
| BatchNorm | **无** | 有 |
| 输出层 bias | **False** | True |
| 架构 | Conv3→ReLU→DConv3(d=2)→ReLU→DConv3(d=4) | Conv3→BN→ReLU→DConv3(d=2)→BN→ReLU→DConv3(d=4)→BN→ReLU→Conv1 |

MVDet 的 BEV head（`map_classifier`）：

```python
nn.Sequential(
    nn.Conv2d(512*7+2, 512, 3, padding=1), nn.ReLU(),
    nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
    nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)
)
```

### 2.3 优化器策略

| 配置 | MVDet 原始 | 我们的实现 |
|------|-----------|-----------|
| 优化器 | **SGD** | Adam |
| 学习率 | **0.1** | 0.0001（差 1000 倍） |
| Momentum | 0.5 | — |
| Weight decay | **5e-4** | 1e-4 |
| Scheduler | **OneCycleLR(max_lr=0.1)** | CosineAnnealingLR |
| Epochs | **10** | 20 |

MVDet 使用高学习率 SGD + OneCycleLR 快速收敛；我们用保守的 Adam + Cosine 衰减。

### 2.4 ResNet-18 空洞卷积

**MVDet 原始**（自定义 resnet.py）：

```python
# BasicBlock — 只 conv1 有 dilation
self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
self.conv2 = conv3x3(planes, planes)  # 无 dilation
```

**我们的实现**：

```python
# _dilate_basic_resnet_layer — conv1 和 conv2 都加了 dilation
block.conv1.dilation = (dilation, dilation)
block.conv2.dilation = (dilation, dilation)  # ← 多余，MVDet 没有
```

多出的 conv2 dilation 改变了 BasicBlock 的感受野模式，可能影响特征质量。

> **注意**：修改 dilation 本身**不会破坏预训练权重**——dilation 和 stride 只是执行参数，不改变权重张量的值。真正的问题是 dilation 施加的位置与 MVDet 不一致。

### 2.5 NMS 后处理

| 参数 | MVDet 原始 | 我们的实现 |
|------|-----------|-----------|
| NMS 方法 | 贪心距离抑制 | 贪心距离抑制 ✅ |
| 抑制半径 | **20 cells = 2.0m** | 5 cells = 0.5m |
| max_preds | **无限** (top_k=50 预排序) | 50 |

MVDet 的 NMS 在 reduced grid 上操作：`dist_thres = 50/2.5 = 20 cells`。每个 reduced cell = 0.1m，所以抑制半径 = 2.0m。

### 2.6 其他次要差异

| 对比项 | MVDet 原始 | 我们的实现 | 影响 |
|--------|-----------|-----------|------|
| 训练集/测试集划分 | 前 90% 训练，后 10% 测试 | 全部帧训练和评估 | 中等 |
| 图像预处理 | Resize(720,1280) + ImageNet normalize | 相同 ✅ | — |
| Grid reduce | 4 | 4 ✅ | — |
| Coord map | 有 | 有 ✅ | — |
| alpha (辅助 loss) | 1.0 | 1.0 ✅ | — |
| Gaussian sigma | 20/4=5.0 | 5.0 ✅ | — |
| Gaussian kernel size | 41 | 41 ✅ | — |

---

## 3. 修复计划

### 修复 1：concat 融合 + MVDet 风格 BEV head

**文件**：`src/models.py`

**目标**：当 `fusion_mode="concat"` 时，BEV head 结构与 MVDet 的 `map_classifier` 完全一致。

**具体改动**：

```python
# 新增 MVDet 风格的 BEV head（替换当前 BEVHeadDilated 用于 concat 模式）
class MVDetMapClassifier(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False),
        )

    def forward(self, x):
        return self.net(x)
```

在 `MVDetLikeNet.__init__` 中，当 `fusion_mode == "concat"` 时使用 `MVDetMapClassifier` 代替 `BEVHeadDilated`。

**验证**：`concat` 模式下 BEV head 输入通道 = `512 * 7 + 2 = 3586`，网络结构与 MVDet 一致。

### 修复 2：优化器改为 SGD + OneCycleLR

**文件**：`src/config.py`、`src/trainer.py`

**具体改动**：

```python
# config.py 新增默认值
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_LR_INIT = 0.1
DEFAULT_MOMENTUM = 0.5
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_SCHEDULER = "onecycle"
DEFAULT_EPOCHS = 10
```

```python
# trainer.py — 构建 optimizer 和 scheduler
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=5e-4)
scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
```

**保留 Adam + Cosine 为 legacy 选项**，可通过 `--optimizer adam --scheduler cosine` 显式选用。

### 修复 3：ResNet-18 dilation 只对 conv1

**文件**：`src/models.py`

**具体改动**：

```python
def _dilate_basic_resnet_layer(layer: nn.Sequential, dilation: int) -> None:
    for block in layer:
        if hasattr(block, "conv1"):
            if block.conv1.stride != (1, 1):
                block.conv1.stride = (1, 1)
            block.conv1.dilation = (dilation, dilation)
            block.conv1.padding = (dilation, dilation)
-       if hasattr(block, "conv2"):
-           block.conv2.dilation = (dilation, dilation)
-           block.conv2.padding = (dilation, dilation)
        if getattr(block, "downsample", None) is not None:
            conv = block.downsample[0]
            if hasattr(conv, "stride") and conv.stride != (1, 1):
                conv.stride = (1, 1)
```

### 修复 4：NMS 半径改为 20 cells

**文件**：`src/evaluate_main.py`、`src/config.py`

**具体改动**：

```python
# config.py
DEFAULT_DET_MIN_DISTANCE = 20.0  # was 5.0, MVDet uses 50/2.5=20 cells

# evaluate_main.py argparse
ap.add_argument("--det_min_distance", type=float, default=DEFAULT_DET_MIN_DISTANCE)
ap.add_argument("--det_max_preds", type=int, default=0)  # 0 = 无限制（MVDet 风格）
```

---

## 4. 修复优先级与依赖

```
修复 1 (concat 融合)  ─┐
修复 2 (SGD+OneCycle) ─┤
修复 3 (dilation fix) ─┼─→ 验证：10 epoch 训练 + eval
修复 4 (NMS 半径)     ─┘
```

所有修复**同时实施**，然后用一次完整训练验证效果。不逐个修复是因为这些差异互相关联——concat 融合需要匹配的 BEV head，SGD 需要匹配的 scheduler，等等。

---

## 5. 验证标准

修复后的第一次训练（concat + SGD + 10 epoch）预期：

| 指标 | 预期范围 | 判断标准 |
|------|----------|----------|
| MODA | > 0.30 | 最低可用 |
| Precision | > 0.50 | — |
| Recall | > 0.40 | — |
| SNR | > 2.0 | 训练过程健康 |
| bev_prediction.png | 高光点与 GT 对齐 | 目视检查 |

如果 MODA 仍低于 0.30，按以下顺序排查：

1. 检查训练 loss 曲线是否正常下降
2. 检查 SNR 走势是否单调上升
3. 检查 bev_prediction.png 热力图
4. 对比 concat 与 confidence_v2 的检测指标
5. 尝试增大 bev_pos_weight

---

## 6. 历史实验对比

| Run | Epoch | 融合 | 优化器 | pos_weight | SNR | MODA | 问题 |
|-----|-------|------|--------|-----------|-----|------|------|
| #41 | 20 | confidence_v2 | Adam 1e-4 | 10.0 | 0.572 | -0.065 | 阈值范围错 |
| #46 | 20 | confidence_v2 | Adam 1e-4 | 10.0 | 0.568 | -0.045 | eval 管线修复后本地跑 |
| **下一次** | **10** | **concat** | **SGD 0.1** | **10.0** | **目标 >2.0** | **目标 >0.30** | 完整对齐 MVDet |

---

## 7. 后续规划

当 concat 基线达到 MODA ≥ 0.30 后：

1. **对比实验**：在相同训练配置下，对比 concat 与 confidence_v2 融合
2. 如果 confidence_v2 不如 concat，分析原因并改进融合模块
3. 如果 concat 仍低于论文水平（0.888），逐步排查：
   - 训练集/测试集划分是否一致
   - 投影矩阵精度
   - 数据增强策略
4. 达到满意效果后，更新 `docs/training_goals.md` 中的进展状态
