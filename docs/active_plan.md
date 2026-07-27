# Active Plan — 当前迭代

> 最后更新：2026-07-27

## 当前状态

### Module 1 🏆 已完成

| 指标 | MVDet baseline | **Ours (best)** | 变化 |
|---|:---:|:---:|:---:|
| MODA | 0.8456 | **0.8950** | **+4.9pp** |
| 参数量 | 32.7M | **5.7M** | **-82.6%** |
| FPS (T4) | 0.62 | **0.96** | **+54.8%** |

### Module 2 进行中

| 组件 | 状态 | 关键结果 |
|---|---|---|
| M2-0 冻结检测器 | ✅ 复核完成 | MODA 0.8634 (cuDNN 差异) |
| M2-1 时序坐标层 | ✅ 代码+测试通过 | 19 tests passed |
| M2-2 Tracking | ✅ **GT 验证通过** | Kalman MOTA=0.939, IDF1=0.969, IDSW=0 |
| M2-3 场映射+基线 | ✅ **基线有数据** | Advection AUPRC=0.7645, Persistence=0.5224 |
| M2-4 ConvLSTM | ❌ **负实验** (原分辨率) | AUPRC=0.0301；bev_down=16 时 AUPRC=0.663 ✅ |
| M2-5 端到端评估 | ✅ **workflow 已修复** | 待触发 L2&L3 首次运行 |
| M2-6 轨迹预测 baseline | ✅ **代码就绪** | 恒速外推 baseline，待 Colab 产出 ADE/FDE |

### 2026-07-27 推进记录

| 任务 | 变更 | 文件 |
|------|------|------|
| P1 修复 workflow 时序 | 下载重试+错误检查、上传校验、restore merge 逻辑 | `.github/workflows/colab-m2-pipeline.yml` |
| P2 补齐 detection_loader 测试 | 18 个测试覆盖 JSONL 加载、位置/分数提取、Hungarian 匹配 | `tests/test_detection_loader.py` (新建) |
| P3 恒速轨迹预测 baseline | 恒速 Kalman 外推 + ADE/FDE 评估框架 | `src/temporal/trajectory_predictor.py` (新建), `tests/test_trajectory_predictor.py` (新建) |
| P4 修复 calibration 不一致 | `CalibrationLoader` 参数化 `intrinsic_subdir`/`extrinsic_subdir` | `src/calibration.py`, `scripts/calibration.py` |

测试结果：122 collected, 121 passed, 1 pre-existing failure (augmentation hflip)

## 阶段：轨迹预测方向决策

### 关键发现
1. **Tracking 非常好**：Kalman+Hungarian 在 GT 检测上 MOTA=0.939、零 ID switch
2. **Field Advection 是极强基线**：AUPRC=0.7645，线性平流已捕获大部分短时运动
3. **ConvLSTM 失败(原分辨率)**：313 训练窗口 + 稀疏 occupancy → 无法超越线性基线
4. **ConvLSTM 可行(低分辨率)**：bev_down=16 (0.4m cell)，AUPRC=0.663 > Advection(0.76 × 1.01)
5. **L2&L3 workflow 已修复**：checkpoint 上传时序问题解决，可立即触发端到端评估

### 下一步（按优先级）
1. **触发 L2&L3 评估**：提交后用 `train_run_id` 参数运行 `colab-m2-pipeline.yml`
2. **产出 ADE/FDE baseline**：在 Colab 上运行 `evaluate_trajectory_baseline(trajectories, split="val")`
3. **方向决策**：根据 ADE/FDE 数值选择 MLP / Social-STGCNN / 深耕场预测
4. **标定（并行）**：由其他人员独立推进枢纽相机外参现场标定
