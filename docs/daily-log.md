# BEV_Track-Predict 每日实验日志

> 时区：UTC+0
> 最新日期在最前面
> 每日记录包含：进展、实验结果、分析、待解决问题

---

## 2026-07-06

### 进展
- 会话恢复，建立规范化每日记录体系
- 确认最新训练结果（run 28560562597）

### 实验结果
| Run ID | 配置 | MODA | Precision | Recall | F1 | MODP |
|--------|------|:----:|:---------:|:------:|:--:|:----:|
| 28560562597 | lr=0.1, 无clip, pos_w=1.0, 360帧训练, eval后40帧 | **0.441** | 0.900 | 0.452 | 0.601 | 0.753 |

### 分析
- 对比之前 MODA=0.57（run 28418202813），下降原因：
  1. 之前训练用了全部 1800 帧（含测试帧，数据泄露），现在正确使用 360 帧
  2. pos_weight 10→1 降低了正样本梯度权重，Recall 从 0.567 下降到 0.452
  3. SNR 仅 0.381，模型可能欠拟合
- MVDet 官方同样用 360 帧训练达到 0.882，说明数据量不是瓶颈
- 所有关键配置已对齐 MVDet，差距根因待排查

### 待解决
- [ ] MODA 0.44 vs 0.88 差距根因排查
- [ ] 考虑增加 epoch 数（10→20/30）
- [ ] 检查两个小差异影响：color jitter、img_head mid_ch

---

## 2026-07-02

### 进展
- 修复 eval frame_start 崩溃（1800→360，WildTrack 只有 400 帧标注）
- 触发首次正确 train/test split 的训练 run

### 实验结果
- Run 28518382970（lr=0.1, 无clip）：训练完成但 eval 崩溃（frame_start=1800 超范围）
- Run 28560562597（修复后）：MODA=0.441（见 07-06 分析）

### 修复
- `scripts/colab_train.py`: frame_start 1800→360, max_frames 200→40, 训练帧上限 1800→360
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
- 详见 `ai_runs/20260701_session_errors/ai_context.md`

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

通过 `scripts/compare_layers.py` 与 MVDet 官方逐层对比，发现 2.5 个月 MODA=0 的根因：

| Bug | 修复 | 影响 |
|-----|------|------|
| BEV H/W 转置 | NB_WIDTH 480→1440, NB_HEIGHT 1440→480 | 整个 BEV 空间宽高反转 |
| GT 坐标映射 | `ix=pos%480, iy=pos//480`, `map_gt[0,ix,iy]` | GT 标签位置错误 |
| Gaussian sigma | MAP_SIGMA 5.0→2.236（√5） | 核宽 5 倍于正确值 |

详见 `ai_runs/20260629_180955/ai_context.md`

---

## 项目里程碑

| 日期 | 里程碑 | MODA |
|------|--------|:----:|
| 05-04 ~ 06-28 | 56 次训练, MODA 始终为 0 | 0.000 |
| 06-29 | 发现 3 个根因 bug | — |
| 06-30 | 首次 MODA > 0 | 0.529 |
| 07-01 | 配置全面对齐 MVDet | — |
| 07-02 | 首次无数据泄露 eval | 0.441 |
| **目标** | **超越 MVDet 基线** | **≥ 0.882** |
