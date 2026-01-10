已完成 ✅

 Wildtrack 标注闭环：positionID -> (ix,iy) -> (x,y) 映射确认，监督点为脚点

 几何验证闭环：完成单相机 warp 到 BEV 的可视化，确认黑边/拉伸为预期现象

 RGB warp 迁移到特征图 warp：ResNet18 stride=8 特征 -> grid_sample 到 BEV，能量图验证路径通

 单相机 POM 训练脚本：可跑通训练并输出 pred/gt 图，基本避免“全 0/灰雾”完全塌缩（但峰仍不稳定）

 关键诊断指标：pos_mse、固定帧可视化等（用于快速判断是否在学习）

本周（第一优先级）🚀：多视角 MVDet-style 最小可训练闭环

交付定义：在 3 视角配置下，fixed_stem 的 pred 在 1~2 小时内能稳定聚集到 gt 团块附近；并能保存模型权重。

 08 多视角脚本跑通（3 views：0,1,2）

 正常打印 valid_ratio / len(loader) 等关键日志

 输出 fixed_stem 的 bev_pred/bev_gt PNG（多个 step）

 保存 model_multicam_mvdet_style.pth

 确认 aux GT 有效（不是空）

 aux_pos_mse 在前 500~1000 step 内显著下降

 若不下降：立即检查 views bbox 解析与坐标缩放

 OOM/速度治理（MX450）

 必要时将 reduce_ch 调至 32

 先去掉 photometric_aug，最后再减少视角

 固定一组“复现实验命令”并写进 README（不可含糊）

下周（第二优先级）：融合质量与评估指标对齐

 扩展视角数：3 -> 5 -> 7（逐步）

 指标升级：增加 Recall@K / Precision@K（比 topK_hit 更稳定）

 引入 offset 回归（可选）：解决峰值落在格子边界导致的对齐抖动

 验证多视角一致增强（view-coherent aug）确实带来稳定收益（A/B）

风险清单（必须盯住）⚠️

 aux bbox 解析错误导致 aux GT 全空：表现为 aux_loss 很小但不降、或 aux_pos_mse=nan

 标定/缩放不一致导致投影 grid 偏移：表现为 pred 峰持续系统性偏移

 过拟合：帧少 + epoch 多导致固定帧看起来很好但泛化差（需要留出验证帧或做简单 hold-out）

决策点（不做无效争论）

如果 3 views + aux loss 仍无法在 2 小时内让 fixed_stem 出现稳定聚集：

先证明 aux GT 真实有效（可视化 head/foot heatmap）

再检查投影 grid 对齐（可视化投影后的有效区域与脚点落点）

仍不行：切换到官方仓库 baseline 做参照
"""