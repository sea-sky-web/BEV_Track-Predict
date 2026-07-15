# Plan: Module 2 实施 — M2-0 到 M2-3

## Context

Module 1（BEV 行人检测）已完成，最佳结果 MODA 0.8950（MobileNet-V2 + geo_confidence_v1，5.7M 参数）。现在需要在冻结检测器基础上推进 Module 2 的前四个阶段：时序坐标层、世界坐标 tracking、时空场映射和非学习预测基线。

详细研究计划见 `docs/module2_spatiotemporal_field_prediction_plan.md`。

### 硬性前置条件

在写任何 M2 代码前，必须先更新三个文档（当前内容明确禁止 tracking/temporal）：

1. **`docs/model_definition.md`** — 扩展阶段边界，增加 M2 范围（tracking、field mapping、prediction）
2. **`docs/dataset_contract.md`** — 增加 `personID` 使用、时序标注、速度目标生成
3. **`docs/m1_frozen_detector_manifest.md`** — 修正 Section 8 的 threshold/NMS 值（0.425/6.0 → 0.375/5.0，与 Section 4 实际最佳结果一致）

---

## 实施阶段

### 阶段 0：文档更新与 Manifest 修正

**修改文件：**
- `docs/model_definition.md` — 新增 Section 11 "Module 2 Stage Boundary"，明确 M2 包含 tracking、field mapping、短时预测，保留 M1 定义不变
- `docs/dataset_contract.md` — 新增 Section 14 "Module 2 Temporal Annotation Contract"，覆盖 personID、时序分割、速度目标、坐标规范
- `docs/m1_frozen_detector_manifest.md` — Section 8 修正为 `threshold=0.375, NMS=5.0`

### 阶段 1：M2-0 检测器封版工具

**新增文件：**
- `scripts/export_detections_jsonl.py` — 加载冻结 checkpoint，对 test split 逐帧推理，提取检测点并转换为世界坐标，输出 JSONL

**JSONL 格式（每行一个检测）：**
```json
{"frame_index": 360, "frame_stem": "00000360", "world_x_m": 1.25, "world_y_m": 3.45, "score": 0.82}
```

> 注：实际运行此脚本需要 checkpoint + GPU（Colab），但脚本本身可先编写和静态验证。

### 阶段 2：M2-1 时序坐标层（`src/temporal/`）

**新增 package `src/temporal/`：**

| 文件 | 职责 |
|---|---|
| `__init__.py` | 包初始化 |
| `annotation_reader.py` | 读取 `annotations_positions/` 全部 JSON，提取 personID + positionID，返回结构化列表 |
| `coordinates.py` | 坐标转换：positionID ↔ full grid ↔ reduced grid ↔ world meters；使用 `src/config.py` 已有常量 |
| `time_utils.py` | 帧排序、frame_index ↔ timestamp_s（2Hz）、时序窗口切割 |
| `schemas.py` | JSONL 点/轨迹记录格式定义、NPZ 场格式定义、读写工具函数 |

**关键设计：**
- `annotation_reader.py` 复用 `config.py` 的 `NB_HEIGHT=480`, `ORIGINE_X_M`, `STEP_M` 等常量
- 不修改现有 `src/dataset.py`，M2 时序读取器独立运作
- 坐标转换统一返回 `(world_x_m, world_y_m)`，内部所有中间坐标带 `_full` 或 `_reduced` 后缀

**新增测试 `tests/test_temporal_coordinates.py`：**
- positionID → full grid → world → full grid 往返精度 < 1e-6
- full grid → reduced grid → world 一致性
- 0.025m (full) vs 0.1m (reduced) 单位正确性
- 边界 positionID（0, 480*1440-1）
- personID 出生、离开、中断、恢复的轨迹提取
- 零行人帧、单行人帧、密集帧
- frame 排序与 2Hz 时间戳

### 阶段 3：M2-2 世界坐标 Tracking（`src/temporal/`）

**新增文件：**

| 文件 | 职责 |
|---|---|
| `tracker_base.py` | 抽象基类 `BaseTracker`，定义 `update(detections) -> tracks` 接口 |
| `tracker_nn.py` | 最近邻 baseline tracker（贪心距离匹配） |
| `tracker_kalman.py` | Kalman + Hungarian tracker：常速度状态 `[x, y, vx, vy]`，`scipy.optimize.linear_sum_assignment` |
| `tracking_metrics.py` | MOTA、IDF1、ID switches、fragmentations 计算；依赖 GT personID |

**Kalman tracker 设计：**
- 状态模型：`s = [x, y, vx, vy]`，常速度转移矩阵 `F(dt=0.5)`
- 观测模型：`H = [[1,0,0,0],[0,1,0,0]]`，只观测位置
- 匹配：世界坐标欧氏距离代价矩阵 + Hungarian
- `min_hits=2`（固定），`max_age` 和 `dist_gate` 为可调参数
- 未匹配检测 → tentative track；连续命中 ≥ min_hits → confirmed
- 超过 max_age 帧无匹配 → terminated

**新增测试 `tests/test_temporal_tracking.py`：**
- 两人交叉轨迹的 ID 保持
- 短时漏检（1-2 帧）后恢复
- 误检不创建长期 track
- track birth 和 death
- MOTA/IDF1 在完美匹配下为 1.0
- 静止目标的 Kalman 预测收敛

### 阶段 4：M2-3 时空场映射与非学习基线（`src/temporal/`）

**新增文件：**

| 文件 | 职责 |
|---|---|
| `field_builder.py` | 从轨迹点构建 occupancy field、velocity field、confidence field、valid_mask |
| `baselines.py` | 四种非学习预测基线：Persistence、Constant Velocity、Field Advection、Oracle |
| `field_metrics.py` | 场预测指标：occupancy AUPRC、velocity EPE (m/s)、轨迹 ADE/FDE |

**Occupancy field：**
```python
O_t(q) = 1 - exp(-sum_i K_sigma(q - p_i_t))
```
高斯核 σ 在 {0.1, 0.2, 0.3}m 中选择（validation）。在 reduced grid (120×360, 0.1m/cell) 上操作。

**Velocity field：**
```python
V_t(q) = sum_i K_sigma(q - p_i_t) * v_i_t / (eps + sum_i K_sigma(q - p_i_t))
```
GT 速度：连续三帧中心差分，首尾前向/后向差分，跨 personID 缺口不插值。

**模型输入张量：** `[occupancy, vx, vy, confidence, valid_mask]`，shape `[5, 120, 360]`

**四种非学习基线：**
1. **Persistence** — 未来 occupancy = 当前 occupancy
2. **Constant Velocity** — 逐轨迹常速度外推后重新栅格化
3. **Field Advection** — 用当前速度场对 occupancy 做半拉格朗日平流
4. **Oracle** — GT identity + GT velocity 构造上界

**新增测试 `tests/test_temporal_fields.py`：**
- 高斯占用峰值位置和幅值
- 空帧 → 零场
- 单行人静止 → velocity field 为零
- 直线运动 → velocity field 方向正确
- valid_mask 与相机覆盖一致
- Persistence baseline = identity
- Constant velocity 在直线运动上精确
- Field advection 质量守恒近似（总 occupancy 变化 < 5%）

---

## 文件变更汇总

**修改（3 个文档）：**
- `docs/model_definition.md`
- `docs/dataset_contract.md`
- `docs/m1_frozen_detector_manifest.md`

**新增（src/temporal/ 包，12 个文件）：**
- `src/temporal/__init__.py`
- `src/temporal/annotation_reader.py`
- `src/temporal/coordinates.py`
- `src/temporal/time_utils.py`
- `src/temporal/schemas.py`
- `src/temporal/tracker_base.py`
- `src/temporal/tracker_nn.py`
- `src/temporal/tracker_kalman.py`
- `src/temporal/tracking_metrics.py`
- `src/temporal/field_builder.py`
- `src/temporal/baselines.py`
- `src/temporal/field_metrics.py`

**新增（脚本，1 个）：**
- `scripts/export_detections_jsonl.py`

**新增（测试，3 个）：**
- `tests/test_temporal_coordinates.py`
- `tests/test_temporal_tracking.py`
- `tests/test_temporal_fields.py`

**不修改的文件：**
- `src/dataset.py`、`src/models.py`、`src/trainer.py`、`src/train_main.py`、`src/evaluate_main.py` — M1 代码不动
- 现有 5 个测试文件 — 必须继续通过

---

## 验证方案

1. **现有测试不回归**：`PYTHONPATH=src pytest tests/test_geometry.py tests/test_metrics.py tests/test_augmentation.py tests/test_smoke_forward.py tests/test_loss.py -v`
2. **新测试全部通过**：`PYTHONPATH=src pytest tests/test_temporal_coordinates.py tests/test_temporal_tracking.py tests/test_temporal_fields.py -v`
3. **静态编译检查**：`python -m compileall src/temporal/ scripts/export_detections_jsonl.py tests/test_temporal_*.py`
4. **M2-0 脚本可 dry-run**：`python scripts/export_detections_jsonl.py --help`（无 checkpoint 不崩溃）

---

## 实施顺序

1. 先更新三个文档 → 用户评审
2. 评审通过后，按 M2-1 → M2-2 → M2-3 顺序实现代码
3. 每个阶段完成后运行对应测试
4. 最后运行全量测试确认无回归
