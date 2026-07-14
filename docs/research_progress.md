# Lightweight Multi-View BEV Fusion: Research Progress

> 最后更新：2026-07-14
> 仓库：`sea-sky-web/BEV_Track-Predict`

---

## 1. 研究动机

MVDet (ECCV 2020) 是多视角 BEV 行人检测的经典方法，在 WildTrack 数据集上报告 MODA 0.882。其核心架构是将 V 个视角的 BEV 特征直接拼接（concat），送入一个大容量 BEV head 做检测。

**问题**：concat 融合的 BEV head 输入通道为 V×512+2 = 3586（7 视角），导致：
- BEV head 参数量 18.9M，占全模型 57.7%
- 参数量和计算量随视角数 V **线性增长**
- 7 视角下全模型 32.7M 参数，不适合边缘部署

**我们的方向**：用轻量的 learned attention fusion 替代暴力 concat，在保持检测精度的前提下大幅降低参数量和计算量。

---

## 2. 方法设计

### 2.1 Learned Attention Fusion (confidence_v2)

替代 MVDet 的 concat 融合：
1. 将 V 个视角的 BEV 特征堆叠为 (B, V, 512, H, W)
2. 用 1×1 Conv 压缩联合表示，预测 per-view softmax 权重
3. 加权融合输出 (B, 512, H, W)

BEV head 输入从 3586ch 降到 514ch（512 + 2 coord），参数量从 18.9M 降到 2.4M。

### 2.2 Geometry-Reliability Prompted Fusion (geo_confidence_v1)

在 confidence_v2 基础上注入静态几何先验：
- valid_mask: BEV cell 是否在相机视野内
- border_margin: 归一化的图像边缘距离
- coverage_count: 多少个视角覆盖该位置

几何分支为融合权重提供空间偏置，帮助模型判断哪个视角更可靠。

### 2.3 MobileNet-V2 轻量 Backbone

将 ResNet-18 (11.2M) 替换为 MobileNet-V2 (3.5M)：
- 使用渐进式 dilation 保持 stride-8 输出分辨率
- features[7:14] 使用 dilation=2（原 stride 16）
- features[14:18] 使用 dilation=4（原 stride 32）
- 1280ch → 1×1 Conv → 512ch，与 ResNet 接口完全一致

---

## 3. 实验结果

### 3.1 检测精度对比（ResNet-18 backbone, 7 views, WildTrack）

统一配置：10 epochs, MSE loss, batch=1, SGD lr=0.1, OneCycleLR, eval: frame_start=360, max_frames=40

| 方法 | fusion_mode | MODA | MODP | Precision | Recall | F1 | TP/FP/FN |
|------|-------------|:----:|:----:|:---------:|:------:|:---:|----------|
| **MVDet baseline** | concat | **0.8456** | 0.7585 | **0.9197** | 0.8897 | **0.9044** | 863/58/89 |
| Learned Attention | confidence_v2 | 0.8277 | 0.7573 | 0.9152 | 0.8729 | 0.8935 | 848/60/104 |
| + Geometry Prior | geo_confidence_v1 | 0.8288 | **0.7669** | 0.9104 | **0.8960** | 0.9031 | 863/74/89 |

- confidence_v2 vs concat: MODA -1.8pp，但参数减半
- geo_confidence_v1: MODP 最佳（+0.8pp），Recall 最高（+0.6pp），定位精度改善

### 3.2 效率对比（T4 GPU, 7 views, 1080×1920）

| Backbone | fusion_mode | 参数量 | 非 BB FLOPs | 延迟 | FPS | vs MVDet |
|----------|-------------|:------:|:-----------:|:----:|:---:|:--------:|
| ResNet-18 | concat (MVDet) | **32.7M** | 1811 GF | **1605ms** | **0.62** | baseline |
| ResNet-18 | confidence_v2 | 16.3M | 390 GF | 1342ms | 0.75 | **+20% FPS** |
| ResNet-18 | geo_confidence_v1 | 16.3M | 390 GF | 1361ms | 0.73 | +18% FPS |
| **MobileNet-V2** | concat | 24.4M | 1811 GF | 1263ms | 0.79 | +27% FPS |
| **MobileNet-V2** | **confidence_v2** | **8.0M** | **390 GF** | **1046ms** | **0.96** | **+55% FPS** |
| **MobileNet-V2** | geo_confidence_v1 | 8.0M | 390 GF | 1044ms | 0.96 | +55% FPS |

**最佳轻量组合 (MobileNet-V2 + confidence_v2)**：
- 参数量：8.0M（MVDet 的 **24.5%**）
- 延迟：1046ms（MVDet 的 **65%**）
- FPS：0.96（MVDet 的 **1.55×**）

### 3.3 可扩展性分析（参数量 vs 视角数）

| 视角数 V | concat 参数 | confidence_v2 参数 | 参数比 | concat FLOPs | cv2 FLOPs | FLOPs 比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 22.1M | 15.2M | 1.5× | 893 GF | 298 GF | 3.0× |
| 7 | 32.7M | 16.3M | 2.0× | 1810 GF | 389 GF | 4.7× |
| 12 | 46.0M | 17.6M | 2.6× | 2957 GF | 503 GF | 5.9× |
| 20 | 67.2M | 19.7M | 3.4× | 4792 GF | 685 GF | 7.0× |

concat 的 BEV head 输入通道 = V×512，参数量线性增长；confidence_v2 的 BEV head 始终 514ch，视角越多优势越大。

---

## 4. 待完成实验

| 实验 | 状态 | GA Run ID |
|------|------|-----------|
| MobileNet-V2 + concat 训练 | running | 29320220026 |
| MobileNet-V2 + confidence_v2 训练 | pending | — |
| MobileNet-V2 精度对比表 | blocked on above | — |
| Focal Loss 消融 | pending | — |
| Offset Head 消融 | pending | — |

---

## 5. 关键结论

1. **Learned attention fusion 是有效的轻量化策略**：用 1.84M 参数的融合模块替代 MVDet 的暴力 concat，BEV head 阶段参数减少 87%、FLOPs 减少 78%，MODA 仅降 1.8pp。

2. **轻量 backbone + 轻量融合的组合效果显著**：MobileNet-V2 + confidence_v2 将全模型参数从 32.7M 压缩到 8.0M（-75.5%），推理速度提升 55%。

3. **几何先验提供互补的定位改善**：geo_confidence_v1 的 MODP 比 confidence_v2 高 1pp，Recall 高 2.3pp，在不增加参数量的前提下改善了定位精度。

4. **融合效率优势随视角数放大**：20 视角场景下，concat 的 FLOPs 是 ours 的 7.0×，参数量是 3.4×。

---

## 6. 项目里程碑

| 日期 | 里程碑 | MODA |
|------|--------|:----:|
| 05-04 ~ 06-28 | 56 次训练, MODA=0 (3 个根因 bug) | 0.000 |
| 06-29 | 修复 BEV H/W 转置、GT 坐标、Gaussian sigma | — |
| 06-30 | 首次 MODA > 0 | 0.529 |
| 07-06 | NMS 半径修复 | 0.793 |
| 07-07 | NMS+阈值网格扫描，pipeline 验证通过 | **0.857** |
| 07-07 | 进入第二阶段：创新超越 MVDet | — |
| 07-13 | fusion_mode 参数化修复，统一对比实验 | — |
| 07-14 | MobileNet-V2 backbone 实现，效率 benchmark | — |
| 07-14 | **论文方向确定：轻量化高效多视角 BEV 融合** | — |

---

## 7. 代码结构

```
src/
  models.py          # ResNet18/50 + MobileNetV2 backbone, 4 种融合模式, BEV head
  geometry.py         # 投影矩阵构建, 透视变换, 几何元数据计算
  train_main.py       # 训练入口
  evaluate_main.py    # 评估入口 (阈值/NMS 网格扫描)

scripts/
  colab_train.py      # Colab 端到端训练+评估脚本
  benchmark_inference.py  # GPU 推理速度 benchmark

.github/workflows/
  colab-train.yml     # 训练 workflow (支持 backbone/fusion_mode/loss_type 参数)
  benchmark.yml       # 推理 benchmark workflow
```
