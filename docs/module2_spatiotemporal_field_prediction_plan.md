# Module 2 Plan — BEV 行人时空场映射与短时预测

> 状态：**仅研究与实施计划，尚未实现**
>
> 创建日期：2026-07-15
>
> 前置模块：WildTrack 多视角 BEV 行人检测
>
> 当前约束：本文档不改变 `docs/model_definition.md` 定义的已实现模型边界，
> 不代表 tracking、轨迹预测或 occupancy-flow 已进入当前代码。

---

## 1. 目标与研究问题

第一模块已经能够把同步多视角图像转换为统一 BEV 平面上的逐帧行人点：

```text
multi-view images
→ BEV heatmap
→ pedestrian points {(row, col, score)}
```

模块二计划在此基础上研究：

```text
逐帧 BEV 点
→ 真实世界坐标
→ 跨帧轨迹关联
→ 占用/速度时空场
→ 未来 0.5s / 1.0s / 2.0s 预测
```

计划采用双层输出：

1. **主要输出**：未来 BEV occupancy field 与 velocity field。
2. **解释输出**：从预测速度场积分得到的个体行人未来轨迹。
3. **诊断输出**：检测、关联、场构建和预测的分层误差。

核心研究问题：

- 当前单帧检测精度是否足以支持稳定的跨帧世界坐标关联？
- 占用场与速度场联合表示是否比纯轨迹集合更适合拥挤、多目标场景？
- 在 WildTrack 有限时序数据上，小型 ConvLSTM 能否稳定超过非学习预测基线？
- 检测误差、ID switch 和预测模型分别贡献了多少端到端误差？

---

## 2. 范围与非目标

### 2.1 计划范围

- 仅使用 WildTrack。
- 使用第一模块冻结的 BEV 点检测结果。
- 使用 `personID` 和 `positionID` 构造 GT 轨迹与预测监督。
- 在统一地面世界坐标系中完成 tracking、field mapping 和 forecasting。
- 建立经典运动基线、场预测基线、小型学习模型和分层评估。

### 2.2 第一版明确不做

- 不联合训练或改写第一模块检测器。
- 不引入跨摄像机 ReID；检测结果已经在全局 BEV 平面融合。
- 不引入 BEVFormer、PETR、LSS、DETR3D 或大型 Transformer。
- 不切换到其他数据集，不宣称跨场景泛化。
- 不进行真实车站部署、拥堵控制或风险决策。
- 不因计划文档而修改当前模型定义；开始写代码前另行评审边界变更。

---

## 3. 当前基础与前置封版门

### 3.1 已具备的基础

当前最佳文档结果为：

| 项目 | 当前最佳结果 |
|---|---:|
| Backbone | MobileNet-V2 truncated |
| Fusion | `confidence_v2` |
| MODA | 0.8918 |
| MODP | 0.7728 |
| Precision | 0.9302 |
| Recall | 0.9097 |
| F1 | 0.9198 |
| 参数量 | 5.7M |

已有几何代码能够完成 full/reduced BEV grid 与世界坐标的转换，评估代码也能从
BEV heatmap 提取带分数的检测点。这些是模块二的直接输入基础。

### 3.2 为什么仍需要 Gate M2-0

GitHub Actions Run `29332987206` 虽然结束状态为 success，但日志显示
`eval_results.json`、`metrics.csv` 和部分可视化未完整下载。因此开始 detector-driven
时序研究前，必须建立一个可独立读回的冻结检测器：

1. 恢复最佳 checkpoint，并计算 SHA256。
2. 固定 commit、backbone、fusion、views、输入尺寸、BEV 分辨率。
3. 固定 threshold `0.425` 与 NMS radius `6.0`。
4. 只执行一次复核评估，归档完整 JSON 和点检测导出。
5. 生成 detector manifest，模块二只通过 manifest 引用检测器。

如果复核无法重现记录结果，允许继续开发 GT 时序链，但不得给出完整端到端结论。
任何 Colab/GPU 运行仍需单独得到用户批准。

---

## 4. 数据可行性与限制

WILDTRACK 提供 7 路同步相机、精确联合标定、400 个 2 Hz 标注帧、超过 300 名
行人，并提供地面位置和身份轨迹。因此它能支持世界坐标 tracking 与短时预测。

但整个标注时长约 200 秒且只有一个场景，意味着：

- 适合验证完整方法链和单场景短时预测。
- 适合经典基线、小型递归模型和严格消融。
- 不适合大型模型，也不足以证明跨场景或部署泛化。

固定时间划分：

| Split | 帧范围 | 帧数 | 用途 |
|---|---:|---:|---|
| Train | 0–319 | 320 | 参数学习 |
| Validation | 320–359 | 40 | 阈值、门控、早停与模型选择 |
| Test | 360–399 | 40 | 一次性最终评估 |

历史 4 帧、未来 4 帧时，可形成约 313 个训练窗口、33 个验证窗口和 33 个测试窗口。
禁止随机拆帧，避免重叠时间窗口泄漏。

论文依据：

- [WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.html)

---

## 5. 坐标与时间契约

### 5.1 完整 BEV 网格

WildTrack 的 `positionID` 解码为：

```text
row_full = positionID mod 480
col_full = positionID div 480

x_m = origin_x + (row_full + 0.5) × 0.025
y_m = origin_y + (col_full + 0.5) × 0.025
```

### 5.2 降采样 BEV 网格

第一模块使用 `bev_down=4`：

```text
shape = 120 × 360
cell_size = 0.1 m

x_m = origin_x + (row_reduced + 0.5) × 0.1
y_m = origin_y + (col_reduced + 0.5) × 0.1
```

`(x_m, y_m)` 是模块二唯一规范坐标。数组 row/column、full grid 和 reduced grid
只能作为带名称、单位和版本的派生字段，禁止使用无单位的 `(x, y)`。

### 5.3 时间定义

```text
frame_index = annotation 文件排序后的连续索引
timestamp_s = frame_index / 2.0
Δt = 0.5 s
```

同时保留原始 `frame_stem`，但不得仅凭文件名数值推断物理时间。

### 5.4 时序记录格式

```text
frame_index
frame_stem
timestamp_s
source              # gt | detector
person_id            # GT 可用，检测输出为空
track_id             # tracking 后生成
world_x_m
world_y_m
score
observed             # 当前状态是否来自真实检测
```

点/轨迹使用 JSONL；稠密场使用压缩 NPZ；配置、坐标版本和 checkpoint 哈希使用 YAML
manifest。所有大文件保存在 `outputs/temporal/<run_id>/`，不得提交到 Git。

---

## 6. Tracking 方案与理论依据

### 6.1 状态模型

每个行人使用常速度状态：

```text
s_t = [x_t, y_t, vx_t, vy_t]ᵀ
s_(t+1) = F(Δt)s_t + w_t
z_t = Hs_t + v_t
```

处理流程：

1. Kalman Filter 预测轨迹下一状态。
2. 使用世界坐标距离或 Mahalanobis distance 建立代价矩阵。
3. 使用 Hungarian algorithm 完成一对一匹配。
4. 未匹配检测创建 tentative track。
5. 连续命中后确认，短时漏检由预测状态维持，超时后终止。

这对应 tracking-by-detection 的经典 SORT 路线：

- [Simple Online and Realtime Tracking, ICIP 2016](https://arxiv.org/abs/1602.00763)

### 6.2 第一版参数选择

- `min_hits=2`。
- `max_age ∈ {1,2,4}`，由 validation IDF1 选择。
- 世界距离 gate `∈ {0.5,0.75,1.0,1.5}m`。
- 选择规则：validation IDF1 优先；IDF1 相同时选择 ID switches 更少者。

第一版不使用 appearance/ReID。如果 Kalman + Hungarian 仍在交叉轨迹中产生明显身份
切换，必须先用 IDSW/fragmentation 证明运动关联不足，再另立 ReID 研究计划。

### 6.3 Tracking 指标

- MOTA、IDF1。
- ID switches、fragmentations。
- FP、FN、轨迹长度分布。
- 可选 HOTA、DetA、AssA，用于分离检测和关联质量。

理论依据：

- [HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking, IJCV 2020](https://arxiv.org/abs/2009.07736)

---

## 7. 时空场定义

### 7.1 Occupancy field

对时刻 `t` 的行人位置 `p_i,t` 使用高斯核栅格化：

```text
O_t(q) = 1 - exp(-Σ_i Kσ(q - p_i,t))
```

其中 `q` 是世界坐标中的 BEV cell center。默认 `σ=0.2m`，在 validation 上比较
`{0.1, 0.2, 0.3}m`。

### 7.2 Velocity field

```text
V_t(q) =
  Σ_i Kσ(q - p_i,t) · v_i,t
  ─────────────────────────
     ε + Σ_i Kσ(q - p_i,t)
```

GT 速度估计：

- 连续三帧使用中心差分。
- 轨迹首尾使用前向/后向差分。
- `personID` 中断超过一帧时不跨缺口插值。

### 7.3 模型输入场

```text
F_t = [occupancy, vx, vy, confidence, valid_mask]
shape = [5, 120, 360]
```

`valid_mask` 来自相机投影覆盖联合区域，训练损失不计算无有效覆盖的 BEV cell。

联合 occupancy + flow 表示同时保留密集场景占用和局部运动方向，其研究基础为：

- [Occupancy Flow Fields for Motion Forecasting, RA-L 2022](https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/)

---

## 8. 非学习预测基线

在训练任何神经网络前，必须完成：

1. **Persistence**：所有未来占用等于当前占用。
2. **Constant Velocity**：逐轨迹常速度外推后重新栅格化。
3. **Field Advection**：用当前速度场对 occupancy 做半拉格朗日平流。
4. **Oracle**：使用 GT identity 与 GT velocity 构造可达到的上界。

这些基线用于回答学习模型是否真正学习了非线性运动，而不是只复现惯性。

---

## 9. 学习模型计划

### 9.1 为什么选择小型 ConvLSTM

ConvLSTM 在输入到状态、状态到状态转移中使用卷积，适合固定二维网格上的时空序列。
相比 Transformer，它的参数量和数据需求更适合当前约 313 个训练窗口。

- [Convolutional LSTM Network, NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)

### 9.2 固定结构

```text
input: 4 historical field frames × 5 channels
encoder: 2-layer ConvLSTM, hidden=32, kernel=3
decoder: 4 parallel convolution heads
output per future step: occupancy_logit + vx + vy
output shape: [4, 3, 120, 360]
```

两层 recurrent 参数约 12 万；单个隐藏状态约 138 万 float。计划先在 T4 上使用
`batch=2` 做显存 smoke test，再决定是否进入正式训练。

### 9.3 损失

```text
L_occ   = 0.5 × weighted_BCE + 0.5 × soft_Dice
L_vel   = occupied-mask SmoothL1
L_trace = occupancy advection consistency

L_total = L_occ + 0.5L_vel + 0.1L_trace
```

`L_trace` 约束预测速度平流后的 occupancy 与下一时刻 occupancy 一致。只允许在 validation
比较 `λ_trace ∈ {0.05,0.1,0.2}`，不得用 test 调参。

### 9.4 个体轨迹输出

从最后一个观测位置开始，在预测速度场中双线性采样并积分：

```text
p̂_i,t+h+1 = p̂_i,t+h + Δt · V̂_t+h(p̂_i,t+h)
```

这样场预测是主要模型，个体轨迹是同一输出的解释结果，不需要在第一版再训练独立的
Social-LSTM。社会交互模型只作为后续比较方向：

- [Social-LSTM, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html)
- [STGAT, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_STGAT_Modeling_Spatial-Temporal_Interactions_for_Human_Trajectory_Prediction_ICCV_2019_paper.html)
- [Trajectron++, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630664.pdf)

Social Force 模型只用于检查碰撞、异常速度和群体运动合理性，不在第一版引入人工力参数：

- [Social Force Model for Pedestrian Dynamics, Physical Review E 1995](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.51.4282)

---

## 10. 分阶段实施计划

### M2-0：冻结检测器

交付 checkpoint manifest、复核指标和带世界坐标的点检测导出。

### M2-1：序列与坐标层

增加独立 temporal annotation reader、世界坐标转换、JSONL/NPZ schema 和测试；不改变
现有 detection dataset 的返回结构。

### M2-2：世界坐标 tracking

完成 nearest-neighbor 与 Kalman + Hungarian，使用 GT 和 detector 两类输入评估。

### M2-3：field mapping 与非学习基线

完成 occupancy/velocity/confidence/valid fields、可视化和三种预测基线。

### M2-4：ConvLSTM

训练 occupancy-only、occupancy+velocity、完整 trace-consistency 三组消融。

固定训练配置：

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch: 2
max_epochs: 100
early_stopping_patience: 10
seeds: [0, 1, 2, 3, 4]
```

### M2-5：端到端评估

分别报告：

1. GT position + GT identity。
2. Detector position + GT association。
3. Detector position + predicted tracking。

该分解用于量化检测、关联与预测分别造成的误差。

---

## 11. 指标、统计与晋级标准

### 11.1 场预测

- Occupancy AUPRC：主指标。
- validation 锁定阈值后的 IoU、Precision、Recall、F1。
- Velocity EPE，单位 `m/s`。
- Flow-warped occupancy IoU。

### 11.2 轨迹预测

- ADE、FDE，单位米。
- 分别报告 0.5s、1.0s、2.0s。
- 分组报告静止、普通运动、拥挤交叉场景。

### 11.3 统计规则

- 5 个随机种子。
- Test 使用长度 4 帧的 moving-block bootstrap，1000 次重采样。
- 报告 mean、standard deviation、95% confidence interval。
- Test 只运行锁定配置一次，不参与选择。

### 11.4 阶段晋级

- M2-0：checkpoint、配置、指标和哈希可独立读回。
- M2-1：坐标、单位、时间和 identity 测试全部通过。
- M2-2：Kalman + Hungarian 的 validation IDF1 优于 nearest-neighbor；若相同则 IDSW 更少。
- M2-3：所有非学习基线均有完整、可复现指标。
- M2-4：完整模型必须超过最佳非学习基线，且配对 block-bootstrap 的 95% CI 不跨零。
- 未超过基线时记录负实验，不继续堆叠更大模型。
- 只有完整 detector-driven 链路验证后，才能宣称模块二有效。

---

## 12. 测试计划

### 12.1 单元测试

- `positionID → full grid → world → grid` 往返。
- full/reduced grid 转换和0.025m/0.1m单位。
- 文件排序、frame stem 与2Hz时间戳。
- `personID` 出生、离开、中断和恢复。
- 静止、直线和转弯轨迹速度。
- 两人交叉、短时漏检、误检、track birth/death。
- 高斯占用峰值、场边界、valid mask。
- 平流方向与质量守恒近似。
- ConvLSTM shape、masked loss、无 NaN。
- 零行人、单行人、密集帧。

### 12.2 集成测试

- 8 帧真实 GT 完成 annotation → track → field → forecast。
- 冻结 checkpoint 完成 images → detections JSONL。
- 完整 detector → tracker → field → forecast。
- 模块二关闭时，现有训练与评估入口行为不变。
- 原 detection tests 必须保持通过。

本地没有真实 WildTrack 和可用 checkpoint 时，只能报告静态/单元测试结果；真实数据结论
必须来自正式远程实验。

---

## 13. 风险与回退

| 风险 | 影响 | 缓解 |
|---|---|---|
| 单场景、样本少 | 过拟合与结论不稳定 | 小模型、短时域、5 seeds、CI、限定结论 |
| 时间泄漏 | 指标虚高 | 固定顺序切分，禁止随机拆帧 |
| 坐标/单位错误 | 全链路失效 | 世界米制为唯一规范，完整往返测试 |
| 检测误差传播 | 无法判断预测模型能力 | GT、GT association、完整 tracker 三级评估 |
| ID switch | 速度场与轨迹污染 | 独立 tracking 指标；ReID 必须由失败触发 |
| 稀疏 occupancy | 学习偏向背景 | weighted BCE、Dice、valid mask |
| Colab 归档失败 | 结果不可复现 | manifest、metrics、error log、artifact readback |

计划实现时使用独立 feature branch 和新增 temporal package。第一模块权重、默认配置、
训练入口和历史 `ai_runs` 不覆盖。任何阶段失败都可放弃模块二分支而不影响检测模块。

---

## 14. 文档与评审门

本文档入库仅表示研究方向和实施顺序被记录，不表示功能已经存在。

开始 M2-0 或任何代码修改前必须：

1. 更新并评审 `docs/model_definition.md`、`docs/dataset_contract.md` 和实验记录协议。
2. 明确前一结果、观测问题、假设、最小改动和目标指标。
3. 按 `docs/rules/review-protocol.md` 完成方案评审。
4. 代码变更后由用户逐项批注，再提交。
5. Colab/GPU 实验单独申请批准。
