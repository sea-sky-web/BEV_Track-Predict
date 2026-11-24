# Handoff

## State
- 项目：`BEV_Track-Predict`，仓库 `sea-sky-web/BEV_Track-Predict`，本地 `/Users/ster/Desktop/fusion/BEV_Track-Predict`
- **Module 1 完成** — MODA 0.8950，参数 5.7M，FPS 0.96
- **Module 2 M2-0~M2-5 全部就绪**，121 个测试通过（122 collected, 1 pre-existing hflip failure）
- M2 三级评估框架已构建（L1/L2/L3），**workflow 时序问题已修复**，待触发首次 L2/L3 运行
- **ConvLSTM 为负实验**（原分辨率 AUPRC=0.03；bev_down=16 时 AUPRC=0.663 ✅）
- **恒速轨迹预测 baseline 已实现**，待 Colab 产出 ADE/FDE 数值
- **标定由其他人员独立推进**，与主线研究隔离

## 2026-07-27 变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `.github/workflows/colab-m2-pipeline.yml` | 修改 | P1: 下载重试+错误检查、上传校验、restore merge |
| `tests/test_detection_loader.py` | 新建 | P2: 18 个测试覆盖 JSONL 加载/位置/分数/匹配 |
| `src/temporal/trajectory_predictor.py` | 新建 | P3: 恒速外推+批量预测+ADE/FDE评估 |
| `tests/test_trajectory_predictor.py` | 新建 | P3: 9 个测试 |
| `src/calibration.py` | 修改 | P4: CalibrationLoader 增加 intrinsic_subdir/extrinsic_subdir |
| `scripts/calibration.py` | 修改 | P4: 同上，统一默认 intrinsic_zero |
| `docs/active_plan.md` | 更新 | 反映 M2-5/M2-6 状态和推进记录 |
| `docs/daily-log.md` | 更新 | 07-27 日志条目 |

## Module 2 实验数据

### Tracking (L1, val split)
| Tracker | MOTA | IDF1 | IDSW |
|---|:---:|:---:|:---:|
| NN | 0.9116 | 0.9577 | 3 |
| **Kalman** | **0.9390** | **0.9691** | **0** |

### 场预测 (L1, val split)
| 方法 | Val AUPRC |
|---|:---:|
| Persistence | 0.5224 |
| **Field Advection** | **0.7645** |
| ConvLSTM (negative) | 0.0301 |

## Code Inventory (src/temporal/)
18 files: __init__.py, coordinates.py, annotation_reader.py, time_utils.py,
schemas.py, tracker_base.py, tracker_nn.py, tracker_kalman.py,
tracking_metrics.py, field_builder.py, baselines.py, field_metrics.py,
detection_loader.py, convlstm.py, temporal_loss.py, temporal_dataset.py,
temporal_trainer.py, train_temporal_main.py, **trajectory_predictor.py** (新增)

## Next
1. ~~MobileNet-V2 eval-only~~ ✅ MODA 0.8445
2. ~~两模型对比表~~ ✅ MobileNet-V2 全面领先 (+4.1pp MODA, 5.7× fewer params)
3. **补完 10 epoch**：ResNet-18 (当前 5) + MobileNet-V2 (当前 9)
4. **多种子**：3-5 seed，报 mean ± std（P1 B2）
5. **M2 重跑**：后向差分常速度 + 统一 AUPRC → MLP vs 常速度结论可能反转
6. **P1 修复**：B3 消融网格、B5 延迟重测、A1 offset head
7. **论文撰写**：待 P0 重跑 + P1 完成后

## Context
- `gh` 需要 `--repo sea-sky-web/BEV_Track-Predict`
- **修正后 checkpoint**:
  - ResNet-18+concat: Run 33735428270 artifact (125MB, 5-epoch, seed=42)
  - MobileNet-V2+geo_cv1: Run 33755384172 artifact (21.7MB, 9-epoch, seed=42)
- **修正后 eval 结果**:
  - ResNet-18: Run 33747560484 → MODA 0.8036, MODP 0.7356, P 0.9682, R 0.8309, F1 0.8943
  - MobileNet-V2: Run 33829519054 → MODA 0.8445, MODP 0.7495, P 0.9094, R 0.9380, F1 0.9235
- **两模型均选出 threshold=0.225, NMS=8.0**
- 旧 checkpoint (main分支, 360帧训练): Run 29345199882 — 不可用于论文
- Colab 训练管线修复: commit 8501bc1 (--branch 参数 + max_frames 320)
- colab download API 无法拉取文件（path 存在但返回 not found），靠 artifact + eval-only 绕过
- colab exec timeout ~2h（10 epoch 需 ~3.5h for MobileNet, ~3.3h for ResNet）
- Google Drive 下载有限流问题（间歇性 403）
- M1 test_augmentation hflip 在 Python 3.14 下有既有 failure
