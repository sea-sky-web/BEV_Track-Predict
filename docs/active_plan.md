# Active Plan — 当前迭代

> 最后更新：2026-07-16

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
| M2-4 ConvLSTM | ❌ **负实验** | AUPRC=0.0301, 远低于基线 |
| M2-5 端到端评估 | 🔧 L2&L3 待修复 | workflow 时序问题 |

## 阶段：Module 2 方向调整

### 关键发现
1. **Tracking 非常好**：Kalman+Hungarian 在 GT 检测上 MOTA=0.939、零 ID switch
2. **Field Advection 是极强基线**：AUPRC=0.7645，线性平流已捕获大部分短时运动
3. **ConvLSTM 失败**：313 个训练窗口 + 稀疏 occupancy (occ_max=0.07) → 学习模型无法超越线性基线

### 待决定的方向
1. **轨迹预测路线**：轻量 GNN (Social-STGCNN) 做个体轨迹预测，WildTrack 稀疏场景更适合
2. **场分辨率调整**：降低网格分辨率（0.3-0.5m cell）增强 occupancy 信号密度
3. **等 MultiviewX 数据集**：在更稠密的合成数据 (~40 人/帧) 上重新验证场预测
4. **完成 L2&L3 三级评估**：修复 workflow checkpoint 上传时序
