# Active Plan — 当前迭代

> 最后更新：2026-07-30

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
| M2-5 端到端评估 | ✅ **L2&L3 完成** | 见下方三级对比表 |
| M2-6 轨迹预测 baseline | ✅ 完成 | ADE=0.155m; MLP 负实验 (0.336m) |
| M2-7 Tracker 优化 | ✅ **网格搜索完成** | max_age=1 → MOTA 0.787→0.822 (+3.5pp) |

### 三级评估对比（Run 30265419077, 2026-07-27）

| Level | 位置来源 | 关联方式 | MOTA | IDF1 | IDSW | Advection AUPRC |
|:---:|---|---|:---:|:---:|:---:|:---:|
| L1 | GT | GT | 0.9390 | 0.9691 | 0 | 0.7645 ±0.13 |
| L2 | Detector | GT | 0.8841 | 0.9410 | 2 | 0.6697 ±0.14 |
| L3 | Detector | Tracker | 0.7866 | 0.9063 | 18 | 0.6550 ±0.15 |

**关键洞察**：
- L1→L2 MOTA 下降 0.055：检测器定位误差导致 FN=66（vs L1 的 28）
- L2→L3 MOTA 下降 0.098：tracker 关联引入 IDSW=18, FP=56（vs L2 的 8）
- Advection AUPRC L1→L3 仅下降 14%（0.7645→0.6550），说明场预测对上游误差有鲁棒性

### 2026-07-30 推进记录

| 任务 | 变更 | 文件 |
|------|------|------|
| 集成轨迹评估 | evaluate_trajectory_baseline() 加入 pipeline | `scripts/run_m2_pipeline.py` |
| 记录 L2/L3 结果 | 三级评估对比表 + 分析 | `docs/active_plan.md`, `docs/daily-log.md` |

## 阶段：轨迹预测方向决策

### 关键发现
1. **Tracking 非常好**：Kalman+Hungarian 在 GT 检测上 MOTA=0.939、零 ID switch
2. **Field Advection 是极强基线**：AUPRC=0.7645，线性平流已捕获大部分短时运动
3. **ConvLSTM 失败(原分辨率)**：313 训练窗口 + 稀疏 occupancy → 无法超越线性基线
4. **ConvLSTM 可行(低分辨率)**：bev_down=16 (0.4m cell)，AUPRC=0.663 > Advection
5. **L2&L3 评估完成**：端到端退化可控，Advection baseline 仍有 0.6550 AUPRC
6. **Tracker 是主要瓶颈**：L2→L3 的 IDSW 增长（2→18）和 FP 增长（8→56）是最大退化来源

### 下一步（按优先级）
1. ~~**触发 L2&L3 评估**~~ ✅ 完成（run 30265419077）
2. ~~**产出 ADE/FDE baseline**~~ ✅ ADE=0.1555m, FDE=0.2693m（run 30510404162）
3. ~~**MLP 轨迹预测**~~ ❌ 负实验（ADE=0.336m, 比 baseline 差 2.2×）
4. ~~**改进 L3 Tracker**~~ ✅ 最优: max_age=1, dist_gate=0.75 → MOTA 0.787→0.822
5. **标定（并行）**：由其他人员独立推进枢纽相机外参现场标定
6. **标定后**：config/dataset 参数化重构 → Hub 场景首次训练
7. **论文撰写**：M1(检测超越 MVDet) + M2(tracking+field+trajectory 完整评估)
