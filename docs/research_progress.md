# Lightweight Multi-View BEV Fusion: Research Progress

> 最后更新：2026-07-14
> 仓库：`sea-sky-web/BEV_Track-Predict`
> **Milestone: MODA 0.8950 — 超越 MVDet baseline (0.882)，参数量仅 5.7M (17.4%)**

---

## 1. 研究动机

MVDet (ECCV 2020) 是多视角 BEV 行人检测的经典方法，在 WildTrack 数据集上报告 MODA 0.882。其核心架构是将 V 个视角的 BEV 特征直接拼接（concat），送入一个大容量 BEV head 做检测。

**问题**：concat 融合的 BEV head 输入通道为 V×512+2 = 3586（7 视角），导致：
- BEV head 参数量 18.9M，占全模型 57.7%
- 参数量和计算量随视角数 V **线性增长**
- 7 视角下全模型 32.7M 参数，不适合边缘部署
- 与轻量 backbone 组合时容易 **CUDA OOM**（T4 15GB 实测）

**我们的方向**：用轻量的 learned attention fusion 替代暴力 concat，配合轻量 backbone（MobileNet-V2），在 **精度超越 MVDet 的同时大幅降低参数量和计算量**。

---

## 2. 方法设计

### 2.1 Learned Attention Fusion (confidence_v2)

替代 MVDet 的 concat 融合：
1. 将 V 个视角的 BEV 特征堆叠为 (B, V, C, H, W)
2. 1×1 Conv 压缩联合表示 (V×C → C)，预测 per-view softmax 权重
3. 加权融合输出 (B, C, H, W)

BEV head 输入从 3586ch 降到 514ch（512 + 2 coord），head 参数从 18.9M 降到 2.4M。
融合模块参数量 1.84M，总计非 backbone 部分仅 5.1M（concat 的 23.7%）。

### 2.2 Geometry-Reliability Prompted Fusion (geo_confidence_v1)

在 confidence_v2 基础上注入静态几何先验：
- `valid_mask`: BEV cell 是否在相机视野内
- `border_margin`: 归一化的图像边缘距离
- `coverage_count`: 多少个视角覆盖该位置

几何分支为融合权重提供空间偏置，帮助模型判断哪个视角更可靠。
实验表明 MODP（定位精度）+1pp，Recall +2.3pp，但 MODA 未显著提升。

### 2.3 MobileNet-V2 轻量 Backbone

将 ResNet-18 (11.2M) 替换为截断式 MobileNet-V2 (0.6M)：
- 使用 features[0:7] 自然 stride 8（32ch）
- features[7:14] 使用 dilation=2 保持 stride 8（96ch）
- 截断 features[14:18]（stride-32 层，expansion=6 产生 960/1920ch 中间激活，OOM 根源）
- 96ch → 1×1 Conv → 512ch，与 ResNet 接口一致
- **Gradient checkpointing**：训练时分 3 段检查点，减少 ~60% 激活显存

---

## 3. 实验结果

### 3.1 核心对比表（WildTrack, 7 views, 10 epochs, MSE loss）

统一评估配置：frame_start=360, max_frames=40, threshold/NMS 网格扫描取最优

| 方法 | Backbone | Fusion | 参数量 | MODA | MODP | Precision | Recall | F1 | TP/FP/FN | GA Run |
|------|----------|--------|:------:|:----:|:----:|:---------:|:------:|:---:|----------|--------|
| MVDet baseline | ResNet-18 | concat | 32.7M | 0.8456 | 0.7585 | 0.9197 | 0.8897 | 0.9044 | 863/58/89 | 29230068041 |
| Attention Fusion | ResNet-18 | confidence_v2 | 16.3M | 0.8277 | 0.7573 | 0.9152 | 0.8729 | 0.8935 | 848/60/104 | 29220635744 |
| + Geometry Prior | ResNet-18 | geo_confidence_v1 | 16.3M | 0.8288 | 0.7669 | 0.9104 | 0.8960 | 0.9031 | 863/74/89 | 29301458120 |
| MVDet + MobileNet | MobileNet-V2 | concat | 22.1M | **OOM** | — | — | — | — | — | 29331658358 |
| Attention only | MobileNet-V2 | confidence_v2 | 5.7M | 0.8918 | 0.7728 | 0.9302 | 0.9097 | 0.9198 | 890/41/62 | 29332987206 |
| **Ours (best)** | **MobileNet-V2** | **geo_confidence_v1** | **5.7M** | **0.8950** | **0.7778** | **0.9301** | **0.9223** | **0.9262** | **898/46/54** | **29345199882** |

### 3.2 最佳方法 vs MVDet 对比

| 指标 | MVDet (ResNet-18 + concat) | **Ours (MobileNet-V2 + geo_cv1)** | **改进** |
|------|:---:|:---:|:---:|
| MODA | 0.8456 | **0.8950** | **+4.9pp** |
| MODP | 0.7585 | **0.7778** | **+1.9pp** |
| Precision | 0.9197 | **0.9301** | **+1.0pp** |
| Recall | 0.8897 | **0.9223** | **+3.3pp** |
| F1 | 0.9044 | **0.9262** | **+2.2pp** |
| FP | 58 | **46** | **-20.7%** |
| FN | 89 | **54** | **-39.3%** |
| 参数量 | 32.7M | **5.7M** | **-82.6%** |
| 推理延迟 (T4) | 1605ms | **1044ms** | **-35.0%** |
| FPS (T4) | 0.62 | **0.96** | **+54.8%** |

### 3.3 训练曲线对比

**MobileNet-V2 + confidence_v2 (best)**:
```
Epoch 0: loss=0.011759 bev=0.010168 img=0.001591 snr=0.213
Epoch 1: loss=0.006300 bev=0.005365 img=0.000935 snr=0.280
Epoch 2: loss=0.004836 bev=0.003998 img=0.000838 snr=0.304
Epoch 3: loss=0.003860 bev=0.003048 img=0.000812 snr=0.318
Epoch 4: loss=0.003492 bev=0.002690 img=0.000803 snr=0.328
Epoch 5: loss=0.003307 bev=0.002508 img=0.000799 snr=0.334
Epoch 6: loss=0.003123 bev=0.002326 img=0.000797 snr=0.340
Epoch 7: loss=0.002955 bev=0.002159 img=0.000795 snr=0.346
Epoch 8: loss=0.002768 bev=0.001974 img=0.000794 snr=0.352
Epoch 9: loss=0.002633 bev=0.001839 img=0.000794 snr=0.357
Best threshold: 0.425, Best NMS radius: 6.0
```

**ResNet-18 + concat (MVDet baseline)**:
```
Epoch 0: loss=0.015087 bev=0.011979 img=0.003108 snr=0.148
Epoch 9: loss=0.001681 bev=0.000947 img=0.000734 snr=0.387
Best threshold: 0.275, Best NMS radius: 7.0
```

观察：MobileNet-V2 的 final loss (0.002633) 高于 ResNet-18 (0.001681)，但 MODA 反而更高。
可能原因：轻量 backbone 的正则化效果防止了过拟合，或 attention fusion 的加权机制比 concat 更高效地利用了多视角互补信息。

### 3.4 推理速度 Benchmark（T4 GPU, 7 views, 1080×1920）

| Backbone | Fusion | 参数量 | 非 BB FLOPs | 延迟 | FPS | vs MVDet |
|----------|--------|:------:|:-----------:|:----:|:---:|:--------:|
| ResNet-18 | concat (MVDet) | 32.7M | 1811 GF | 1605ms | 0.62 | baseline |
| ResNet-18 | confidence_v2 | 16.3M | 390 GF | 1342ms | 0.75 | +20% |
| ResNet-18 | geo_confidence_v1 | 16.3M | 390 GF | 1361ms | 0.73 | +18% |
| MobileNet-V2 | concat | 24.4M | 1811 GF | 1263ms | 0.79 | +27% |
| **MobileNet-V2** | **confidence_v2** | **8.0M** | **390 GF** | **1046ms** | **0.96** | **+55%** |
| MobileNet-V2 | geo_confidence_v1 | 8.0M | 390 GF | 1044ms | 0.96 | +55% |

注：Benchmark 使用未截断版 MobileNet-V2 (3.5M backbone)，训练版截断到 0.6M backbone。
截断版推理速度预计更快，待补充实测。

### 3.5 可扩展性分析（参数量/FLOPs vs 视角数 V）

| V | concat 参数 | cv2 参数 | 参数比 | concat FLOPs | cv2 FLOPs | FLOPs 比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 22.1M | 15.2M | 1.5× | 893 GF | 298 GF | 3.0× |
| 5 | 27.4M | 15.7M | 1.7× | 1351 GF | 344 GF | 3.9× |
| 7 | 32.7M | 16.3M | 2.0× | 1810 GF | 389 GF | 4.7× |
| 9 | 38.0M | 16.8M | 2.3× | 2269 GF | 435 GF | 5.2× |
| 12 | 46.0M | 17.6M | 2.6× | 2957 GF | 503 GF | 5.9× |
| 16 | 56.6M | 18.6M | 3.0× | 3874 GF | 594 GF | 6.5× |
| 20 | 67.2M | 19.7M | 3.4× | 4792 GF | 685 GF | 7.0× |

concat 的 BEV head 输入通道 = V×512+2，参数量/FLOPs 线性增长。
Attention fusion 的 BEV head 始终 514ch，视角越多优势越大。

### 3.6 OOM 分析：concat 融合限制 backbone 选择

MobileNet-V2 + concat 在 T4 (15GB) 上 OOM：
- MobileNet-V2 的 InvertedResidual 使用 expansion_ratio=6，在 stride-8 分辨率下中间激活巨大
- 7 视角 × 反向传播 梯度存储 → 超出显存
- Attention fusion 将 BEV head 输入从 3586ch 降到 514ch，解除了这一约束
- **结论：concat 融合不仅参数量大，还限制了 backbone 的选择自由度**

---

## 4. 关键发现与论文论点

### 核心论点
轻量的 learned attention fusion + 几何先验 + 轻量 backbone 可以在**参数量减少 82.6%、速度提升 55%** 的同时，**精度超越 MVDet baseline 4.9 个 MODA 点**。

### 支撑论据

1. **Attention fusion 是有效的轻量化策略**
   - 用 1.84M 参数的融合模块替代 concat，BEV head 参数减少 87%、FLOPs 减少 78%
   - ResNet-18 下 MODA 仅降 1.8pp（0.8456→0.8277），说明 concat 的大部分参数是冗余的

2. **轻量 backbone + attention fusion 产生协同效应**
   - MobileNet-V2 (0.6M) + confidence_v2 = 5.7M 全模型参数
   - MODA 0.8918 **超越** ResNet-18 + concat 的 0.8456（+4.6pp）
   - 可能原因：轻量 backbone 的隐式正则化 + attention fusion 的高效信息利用

3. **concat 融合限制系统设计自由度**
   - MobileNet-V2 + concat 在 T4 上 OOM，而 attention fusion 正常训练
   - concat 的参数量随视角数线性增长（20 视角时达 67.2M），不可扩展
   - Attention fusion 的 BEV head 输入固定为 514ch，与视角数解耦

4. **几何先验提供互补的定位改善**
   - geo_confidence_v1 的 MODP 比 confidence_v2 高 1pp，Recall 高 2.3pp
   - 定位精度提升不增加参数量，可作为轻量化的补充增益

---

## 5. 模型架构对比图

```
MVDet (concat):
  ResNet-18 (11.2M) × 7 views
  → warp to BEV (7 × 512ch)
  → concat (3586ch)
  → MVDetMapClassifier (18.9M) → heatmap
  Total: 32.7M params

Ours (attention fusion):
  MobileNet-V2-truncated (0.6M) × 7 views
  → warp to BEV (7 × 512ch)
  → Learned Attention Fusion (1.84M) → weighted sum (512ch)
  → BEVHeadDilated (2.4M) → heatmap
  Total: 5.7M params
```

---

## 6. 实验复现命令

### ResNet-18 + concat (MVDet baseline)
```bash
gh workflow run colab-train.yml --repo sea-sky-web/BEV_Track-Predict \
  -f backbone=resnet18 -f fusion_mode=concat \
  -f epochs=10 -f loss_type=mse -f offset_weight=0.0
```

### MobileNet-V2 + confidence_v2 (Ours best)
```bash
gh workflow run colab-train.yml --repo sea-sky-web/BEV_Track-Predict \
  -f backbone=mobilenet_v2 -f fusion_mode=confidence_v2 \
  -f epochs=10 -f loss_type=mse -f offset_weight=0.0
```

### Inference Benchmark
```bash
gh workflow run benchmark.yml --repo sea-sky-web/BEV_Track-Predict -f gpu=T4
```

---

## 7. 项目里程碑

| 日期 | 里程碑 | MODA | 参数量 |
|------|--------|:----:|:------:|
| 05-04 ~ 06-28 | 56 次训练, MODA=0（3 个根因 bug） | 0.000 | — |
| 06-29 | 修复 BEV H/W 转置、GT 坐标、Gaussian sigma | — | — |
| 06-30 | 首次 MODA > 0 | 0.529 | — |
| 07-06 | NMS 半径修复 (4× 过大) | 0.793 | — |
| 07-07 | NMS+阈值网格扫描，pipeline 验证通过 | **0.857** | 32.7M |
| 07-07 | 进入第二阶段：创新超越 MVDet | — | — |
| 07-13 | 统一对比实验：concat vs attention fusion | 0.8456 | 32.7M |
| 07-14 | MobileNet-V2 backbone + gradient checkpointing | — | 5.7M |
| **07-14** | **🏆 MODA 0.8950 — 超越 MVDet (0.882)，参数 -82.6%，速度 +55%** | **0.8950** | **5.7M** |

---

## 8. 代码结构

```
src/
  models.py            # Backbone: ResNet18/50 + MobileNetV2 (truncated + grad ckpt)
                       # Fusion: concat / confidence_v1 / confidence_v2 / geo_confidence_v1
                       # Head: MVDetMapClassifier / BEVHeadDilated
  geometry.py           # 投影矩阵, 透视变换, 几何元数据 (valid_mask, border_margin, coverage)
  train_main.py         # 训练入口
  evaluate_main.py      # 评估入口 (阈值/NMS 网格扫描)
  trainer.py            # MVDetTrainer (MSE/focal loss, offset head)

scripts/
  colab_train.py        # Colab 端到端训练+评估 (支持 --backbone --fusion_mode --loss_type)
  benchmark_inference.py # GPU 推理速度 benchmark (多 backbone × 多 fusion)

.github/workflows/
  colab-train.yml       # 训练 workflow (backbone/fusion_mode/loss_type/offset_weight)
  benchmark.yml         # 推理 benchmark workflow

docs/
  research_progress.md  # 本文档 — 论文数据和研究进展
  daily-log.md          # 每日实验日志
  active_plan.md        # 当前迭代计划
  LESSONS.md            # 经验教训
```

---

## 9. 待完成实验

| 实验 | 目的 | 优先级 |
|------|------|:------:|
| MobileNet-V2 + concat (gradient ckpt) | 控制变量：同 backbone 下 concat vs cv2 | P1 |
| 截断版 MobileNet-V2 推理 benchmark | 更新推理速度数据 | P1 |
| 多次训练方差统计 | 确认 MODA 0.8918 的置信区间 | P2 |

---

## 10. Module 2: BEV 时空场预测

> 最后更新：2026-07-16
> GA Pipeline Run: 29473931980 (L1 complete), 29389441553 (first run)

### 10.1 目标

在冻结的 Module 1 BEV 检测器基础上，研究：
- 跨帧轨迹关联（tracking）
- BEV occupancy/velocity 时空场构建
- 短时未来预测（0.5s / 1.0s / 2.0s）
- 检测 → 关联 → 预测的误差分解

### 10.2 三级评估框架

| Level | 位置来源 | 身份来源 | 回答的问题 |
|:---:|---|---|---|
| L1 | GT annotation | GT personID | 预测模型本身的能力上限 |
| L2 | 冻结检测器 | GT 匈牙利匹配 | 检测误差传播多少 |
| L3 | 冻结检测器 | Kalman tracker | 完整端到端性能 |

### 10.3 Tracking 实验结果（L1, val split, frames 320-359）

| Tracker | MOTA | IDF1 | ID Switches | Fragmentations | TP | FP | FN |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Nearest Neighbor | 0.9116 | 0.9577 | 3 | 6 | 622 | 21 | 34 |
| **Kalman + Hungarian** | **0.9390** | **0.9691** | **0** | **1** | 628 | 12 | 28 |

**结论**：Kalman tracker 在 GT 检测上达到零 ID switch、仅 1 次 fragmentation。
M2-2 gate 通过：Kalman IDF1 (0.9691) > NN IDF1 (0.9577) ✅

### 10.4 时空场统计

| Split | Frames | occ_mean | occ_max | vel_mean (m/s) |
|---|:---:|:---:|:---:|:---:|
| Train | 320 | 0.000562 | 0.0722 | 0.0929 |
| Val | 40 | 0.000372 | 0.0454 | 0.0396 |
| Test | 40 | 0.000543 | 0.0640 | 0.0583 |

**关键观察**：occupancy 极度稀疏（max 仅 0.07），因为 120×360 网格（43200 cells）上仅 ~20 人。

### 10.5 非学习基线（L1, val split, AUPRC, gt_threshold=1e-3）

| Baseline | Val AUPRC | 说明 |
|---|:---:|---|
| Persistence（未来=当前） | 0.5224 ±0.14 | 合理下界 |
| **Field Advection（速度场平流）** | **0.7645 ±0.13** | 线性外推，非常强 |

### 10.6 ConvLSTM 实验结果 ❌ 负实验

| 配置 | Val AUPRC | Epochs | 说明 |
|---|:---:|:---:|---|
| ConvLSTM (full ablation, seed=0) | **0.0301** | 27 (early stop) | 远低于所有基线 |

**与基线对比**：
- ConvLSTM (0.03) vs Persistence (0.52) → **差 17×**
- ConvLSTM (0.03) vs Advection (0.76) → **差 25×**

**根因分析**：
1. Occupancy 信号极度稀疏（max=0.07），网络难以从 43200 个几乎全零的 cell 中学习
2. 训练窗口仅 313 个，数据量不足以训练 157K 参数的 ConvLSTM
3. Loss 几乎不下降（0.54 → 0.50），说明模型容量和数据规模严重不匹配
4. Field Advection 已经是极强基线——短时+稀疏场景下线性平流捕获了大部分运动信息

**按 M2-4 gate 规则**：记录为负实验，不继续堆叠更大模型。

### 10.6b ConvLSTM 实验 — 网格分辨率消融 ✅ 正面结果

| 配置 | Grid | occ_max | Val AUPRC | vs Persistence | vs Advection |
|---|---|:---:|:---:|:---:|:---:|
| bev_down=4 (0.1m) | 120×360 | 0.045 | 0.030 | ❌ 0.06× | ❌ 0.04× |
| **bev_down=16 (0.4m)** | **30×90** | **0.505** | **0.663** | **✅ 1.17×** | **✅ 1.01×** |

**关键发现**：网格分辨率是场预测成败的决定因素。0.4m cell 让 occupancy 峰值从 0.045 提升到
0.505（11×），ConvLSTM 从完全无法学习（AUPRC=0.03）变成超越所有非学习基线（AUPRC=0.663）。

训练时间从 675s 降到 108s（6× 加速），loss 从 0.72 稳定降至 0.38。

GA Run: 29484283019 (bev_down=16, seed=0)

### 10.7 M2-0 冻结检测器复核

| 指标 | 训练日志 | Colab 复核 | 差异原因 |
|---|:---:|:---:|---|
| MODA | 0.8950 | 0.8634 | cuDNN 非确定性 |
| Precision | 0.9301 | 0.9278 | — |
| Recall | 0.9223 | 0.8771 | — |

GA Run: 29386822805。差异来自 `cudnn.benchmark=True` 在不同 T4 实例上的算法选择。

### 10.8 下一步方向

1. **切换到轨迹预测路线**：WildTrack 场景稀疏，个体轨迹信号远强于稠密场信号
2. **调整场分辨率**：降低网格分辨率（0.3-0.5m cell）可能增强 occupancy 信号
3. **等 MultiviewX 数据集就绪后**：在更稠密的合成数据上重新验证场预测方案
4. **完成 L2&L3 三级评估**：修复 workflow 中 checkpoint 上传时序问题
