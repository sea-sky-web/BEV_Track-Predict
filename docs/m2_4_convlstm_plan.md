# Plan: M2-4 ConvLSTM 时空场预测模型

## Context

Module 2 前三阶段（坐标层、tracking、场映射与非学习基线）已完成并通过 49 个测试。现在实现 M2-4：ConvLSTM 时空场预测模型。该模型接收 4 帧历史 BEV 场（occupancy + velocity + confidence + valid_mask），预测未来 4 帧的 occupancy 和 velocity field。

按照 `docs/module2_spatiotemporal_field_prediction_plan.md` Section 9-10 的设计。

---

## 新增文件

### 1. `src/temporal/convlstm.py` — ConvLSTM Cell + 预测模型

**ConvLSTMCell**：标准 ConvLSTM 单元
- 输入：`(B, C_in, H, W)` + 上一步 hidden `(h, c)`
- 输出：新的 `(h, c)`，shape `(B, C_hidden, H, W)`
- 参数：`in_channels`, `hidden_channels`, `kernel_size`, padding='same'

**SpatioTemporalPredictor (nn.Module)**：完整预测模型
- Encoder：2 层 ConvLSTM，hidden=32，kernel=3
- Decoder：每个 future step 一个共享的 `Conv2d` head，输出 3 通道 (occ_logit, vx, vy)
- forward 签名：
  ```python
  def forward(self, x: Tensor) -> Tensor:
      # x: (B, T_hist=4, C_in=5, H=120, W=360)
      # return: (B, T_future=4, 3, H, W)
  ```
- Decoder 逐步解码：使用 encoder 最终 hidden state，每步通过 ConvLSTM decoder layer 生成 hidden，再经 output conv 得到 (occ_logit, vx, vy)

### 2. `src/temporal/temporal_loss.py` — 损失函数

三个损失组件：

- **OccupancyLoss**：`0.5 * weighted_BCE(sigmoid(logit), gt) + 0.5 * soft_Dice(sigmoid(logit), gt)`
  - valid_mask 控制哪些 cell 参与 loss
  
- **VelocityLoss**：`SmoothL1(pred_v, gt_v)` 仅在 `gt_occupancy > 0.5` 的位置计算

- **TraceConsistencyLoss**：用预测 velocity 对预测 occupancy 做半拉格朗日平流，与下一步预测 occupancy 比较
  - 复用 `baselines.py` 中 advection 的数学逻辑，但用 PyTorch 实现（支持梯度）

- **CombinedTemporalLoss**：
  ```
  L = L_occ + lambda_vel * L_vel + lambda_trace * L_trace
  ```
  默认 `lambda_vel=0.5`, `lambda_trace=0.1`

### 3. `src/temporal/temporal_dataset.py` — 时序场数据集

**FieldSequenceDataset (torch.utils.data.Dataset)**：
- 初始化时：接收 annotation_dir + split name，构建所有帧的 field（调用 `build_all_fields`），缓存为内存数组
- `__getitem__(idx)` 返回：
  - `history_fields`: `(T_hist, 5, H, W)` float32 tensor
  - `future_fields`: `(T_future, 5, H, W)` float32 tensor（GT target）
- 使用 `time_utils.make_temporal_windows` 生成窗口索引
- 支持 `sigma_m` 参数用于 occupancy kernel 选择

### 4. `src/temporal/temporal_trainer.py` — 训练器

沿用 M1 `MVDetTrainer` 的模式：
- `train_epoch(loader, epoch) -> dict`
- `validate(loader) -> dict`：返回 val_loss + occupancy_auprc
- `save_checkpoint(epoch, metrics)`
- Early stopping：patience=10，监控 val occupancy AUPRC
- Seed 设置：`torch.manual_seed`, `np.random.seed`, `torch.backends.cudnn.deterministic=True`

### 5. `src/temporal/train_temporal_main.py` — 训练入口

CLI 入口，参数：
- `--annotations_dir`：WildTrack annotations 路径
- `--output_dir`：输出目录
- `--seed`：随机种子（运行 5 次用 0-4）
- `--epochs`、`--batch`、`--lr`、`--weight_decay`
- `--lambda_vel`、`--lambda_trace`：损失权重
- `--sigma_m`：occupancy kernel sigma
- `--history_len`、`--future_len`
- `--ablation`：`occ_only` | `occ_vel` | `full`（三组消融）
- `--device`

### 6. `tests/test_temporal_convlstm.py` — 测试

- ConvLSTM cell forward shape 正确
- SpatioTemporalPredictor forward shape `(B, 4, 3, 120, 360)`
- backward 梯度有限
- OccupancyLoss 对完美预测返回接近零
- VelocityLoss 仅计算 occupied 区域
- TraceConsistencyLoss 有限且非零
- CombinedTemporalLoss 三种 ablation 模式
- FieldSequenceDataset 返回正确 shape
- masked loss 无 NaN
- 空帧不崩溃

---

## 关键设计决策

1. **Decoder 结构**：采用 autoregressive decoding — encoder 处理完历史序列后，decoder ConvLSTM 逐步生成未来帧。每步输出经 conv head 得到 (occ, vx, vy)，同时该输出拼上 valid_mask 和 confidence 后作为下一步 decoder 输入。这样未来第 2 步可以利用第 1 步的预测。

2. **数据预处理策略**：dataset 初始化时一次性构建所有帧的 field 并缓存（400 帧 × 5 × 120 × 360 × 4bytes ≈ 345MB），避免训练时反复计算高斯核。

3. **advection consistency 的 PyTorch 实现**：使用 `F.grid_sample` 实现可微分的半拉格朗日平流，使 L_trace 可以反向传播梯度。

4. **不修改任何 M1 代码**：所有新文件在 `src/temporal/` 下。

---

## 文件变更汇总

**新增 5 个文件：**
- `src/temporal/convlstm.py`
- `src/temporal/temporal_loss.py`
- `src/temporal/temporal_dataset.py`
- `src/temporal/temporal_trainer.py`
- `src/temporal/train_temporal_main.py`

**新增 1 个测试文件：**
- `tests/test_temporal_convlstm.py`

**不修改任何现有文件。**

---

## 验证方案

1. **编译检查**：`python3 -m compileall src/temporal/convlstm.py src/temporal/temporal_loss.py src/temporal/temporal_dataset.py src/temporal/temporal_trainer.py src/temporal/train_temporal_main.py`
2. **新测试通过**：`PYTHONPATH=src python3 -m pytest tests/test_temporal_convlstm.py -v`
3. **现有测试不回归**：`PYTHONPATH=src python3 -m pytest tests/ -v`
4. **dry-run**：`PYTHONPATH=src python3 src/temporal/train_temporal_main.py --help`
5. **实际训练需要 Colab**（WildTrack 数据 + GPU），单独申请批准
