# BEV_Track-Predict 每日实验日志

> 时区：UTC+0
> 最新日期在最前面
> 每日记录包含：进展、实验结果、分析、待解决问题

---

## 2026-07-06

### 进展
- 会话恢复，建立规范化每日记录体系
- 深入排查 MODA 0.44 vs 0.88 差距根因
- 发现 **3 个被遗漏的致命问题** + 2 个中等问题

### 深入排查发现

#### P0（致命）：Workflow 默认值从未与代码同步
| 参数 | workflow 默认值 | 代码默认值 | MVDet 值 |
|------|:-----------:|:-------:|:------:|
| max_frames | **100** | -1(全部) | 360 |
| bev_pos_weight | **10.0** | 1.0 | 1.0 |

workflow 直接传参覆盖代码默认值 → 训练实际用 100 帧 + 10 倍正样本权重。

#### P1：AMP 混合精度 + Color Jitter
- `colab_train.py` 硬编码 `--amp`：MVDet 不用，float16 精度不足 + OneCycleLR 失步
- `--augment true` + color_jitter(0.2,0.2,0.2,0.05)：MVDet 无任何增强

#### P2：Backbone dilation 实现不一致
- 我们对 BasicBlock conv1 强制加 dilation=2/4
- MVDet 旧版 torchvision BasicBlock 忽略 dilation，只改 stride=1
- dilation 破坏 ImageNet 预训练权重的 3×3 连续感受野

### 修复（PR fix/align-training-config-p0p1p2）
| 文件 | 修改 |
|------|------|
| `.github/workflows/colab-train.yml` | max_frames 100→360, bev_pos_weight 10.0→1.0 |
| `scripts/colab_train.py` | max_frames 默认 100→360, 移除 --amp, augment true→false |
| `src/models.py` | 移除 dilation, 只保留 stride=1（_undilate_basic_resnet_layer） |

### 实验结果
| Run ID | 配置 | MODA | Precision | Recall | F1 | MODP |
|--------|------|:----:|:---------:|:------:|:--:|:----:|
| 28560562597 | lr=0.1, 无clip, pos_w=1.0, 360帧训练, eval后40帧 | **0.441** | 0.900 | 0.452 | 0.601 | 0.753 |

### 分析
- MODA 0.44 的主因大概率是 workflow 默认值（100 帧 + pos_w=10.0）
- AMP、color jitter、backbone dilation 为次要因素
- 修复后需要重新训练验证

### 待解决
- [ ] 修复推送并经过 CR 后合并
- [ ] 触发新训练 run 验证 MODA 提升
- [ ] 若 MODA 仍不足，考虑增加 epoch（10→20）或输入分辨率（720×1280→1080×1920）

### NMS 半径 bug（07-06 晚间追加）

**根因确认**：`evaluate_main.py` 的 `det_min_distance=20.0` 直接用在 REDUCED grid 上，而 MVDet 的 `nms(dist_thres=20)` 用在 FULL grid 上。

| | MVDet | 我们 |
|---|---|---|
| NMS 参数 | 20 | 20 |
| 坐标系 | Full grid (2.5cm/格) | Reduced grid (10cm/格) |
| 实际抑制半径 | **0.5m** | **2.0m（4× 过大）** |

修复：`det_min_distance` 20.0 → 5.0。此 bug 直接解释 Recall=0.456（FN=503/952）。

### 验证结果（L4 run 28790760800）

| 指标 | 修复前 | NMS 修复后 | MVDet 目标 |
|---|:---:|:---:|:---:|
| **MODA** | 0.441 | **0.793** | 0.882 |
| Precision | 0.914 | 0.886 | — |
| Recall | 0.456 | **0.900** | — |
| F1 | 0.608 | **0.893** | — |
| TP/FP/FN | 449/26/503 | 861/106/91 | — |

NMS 根因确认：Recall 翻倍（0.456→0.900），FN 减少 82%（503→91）。

### 待解决
- [ ] MODA 0.793 vs 0.882，差距 0.089 主要来自 FP=106
- [ ] Google Drive 持久化已合并（PR #74），下次 run 可验证 checkpoint 下载

---

## 2026-07-02

### 进展
- 修复 eval frame_start 崩溃（1800→360，WildTrack 只有 400 帧标注）
- 触发首次正确 train/test split 的训练 run

### 实验结果
- Run 28518382970（lr=0.1, 无clip）：训练完成但 eval 崩溃（frame_start=1800 超范围）
- Run 28560562597（修复后）：MODA=0.441（见 07-06 分析）

### 修复
- [`scripts/colab_train.py`](../scripts/colab_train.py): frame_start 1800→360, max_frames 200→40, 训练帧上限 1800→360
- 移除重复的 `--max_frames` 参数（CR 发现）

---

## 2026-07-01

### 进展
- 端到端 forward pass 对比完成（E/F/G 节点）
- 读取 MVDet 官方训练配置（main.py, trainer.py, frameDataset.py）
- 识别并修复 3 个训练配置差异：lr, grad_clip, bev_pos_weight
- L4 GPU smoke test 通过（session 稳定可用）
- A100 确认不可用（免费 Colab 账号）

### 关键发现
- **BEV head 完全一致**：in_ch=3586, 3层 dilated 512ch, bias=False
- **img_head 有差异**：我们 Conv2d(512,128,3×3) vs MVDet Conv2d(512,64,1×1)（辅助任务，影响小）
- **MVDet 官方配置**：lr=0.1, momentum=0.5, wd=5e-4, epochs=10, OneCycleLR, 无 grad_clip, 无 augmentation, train_ratio=0.9
- **我们此前的差异**：lr=0.05（差2倍）, grad_clip=1.0（MVDet无）, bev_pos_weight=10（MVDet=1）

### 修复（通过 PR）
| PR | 修复内容 |
|----|----------|
| #66 | lr 0.05→0.1, 移除 grad_clip, eval 改后 200 帧 |
| #68 | frame_start 1800→360（WildTrack 400帧） |
| #69 | bev_pos_weight 10→1 |

### 错误记录
- 详见 [`ai_runs/20260701_session_errors/ai_context.md`](../ai_runs/20260701_session_errors/ai_context.md)

---

## 2026-06-30

### 进展
- 逐层对比验证全部 PASS（修复后）
- 首次全量训练（1800帧 × 10 epochs）
- BEV 可视化脚本重写（raw logit 归一化, 3x 放大）
- base64 导出机制建立

### 实验结果
| Run ID | 配置 | MODA | 备注 |
|--------|------|:----:|------|
| 28364780247 | 逐层对比 | ALL PASS | 投影矩阵/coord/GT/kernel |
| 28364788716 | 200帧, lr=0.05, pos_w=10 | 0.529 | 首次 MODA>0 |
| 28418202813 | 1800帧, lr=0.05, pos_w=10 | 0.572 | 含测试帧（数据泄露） |

---

## 2026-06-29

### 重大进展：发现并修复 3 个根因 bug

通过 [`scripts/compare_layers.py`](../scripts/compare_layers.py) 与 MVDet 官方逐层对比，发现 2.5 个月 MODA=0 的根因：

| Bug | 修复 | 影响 |
|-----|------|------|
| BEV H/W 转置 | NB_WIDTH 480→1440, NB_HEIGHT 1440→480 | 整个 BEV 空间宽高反转 |
| GT 坐标映射 | `ix=pos%480, iy=pos//480`, `map_gt[0,ix,iy]` | GT 标签位置错误 |
| Gaussian sigma | MAP_SIGMA 5.0→2.236（√5） | 核宽 5 倍于正确值 |

详见 [`ai_runs/20260629_180955/ai_context.md`](../ai_runs/20260629_180955/ai_context.md)

---

## 项目里程碑

| 日期 | 里程碑 | MODA |
|------|--------|:----:|
| 05-04 ~ 06-28 | 56 次训练, MODA 始终为 0 | 0.000 |
| 06-29 | 发现 3 个根因 bug | — |
| 06-30 | 首次 MODA > 0 | 0.529 |
| 07-01 | 配置全面对齐 MVDet | — |
| 07-02 | 首次无数据泄露 eval | 0.441 |
| 07-06 | 深入排查：发现 workflow 默认值、AMP、dilation 3 个遗漏问题 | — |
| 07-06 | **根因确认：NMS 半径 4× 过大**（20 reduced cells=2.0m，应为 5 cells=0.5m） | — |
| **07-06** | **🏷️ v0.1.0-moda79 — MODA 0.793** | **0.793** |
| **目标** | **超越 MVDet 基线** | **≥ 0.882** |
