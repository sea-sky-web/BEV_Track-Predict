# Exploration Memory

本文档用于沉淀项目在 Wildtrack 多视角 BEV 检测中的关键探索，避免历史结论在后续模型构建中丢失。

## Phase 1: 标注与坐标闭环
- 问题：`positionID` 与 BEV 网格/世界坐标映射是否正确。
- 尝试：在 `02_draw_bboxes.py`、`03_geom_closure_validate.py` 中验证 `positionID -> (ix,iy) -> (x,y)`，并与投影结果比对。
- 结论：脚点监督语义成立；`positionID` 网格映射可作为主监督来源。
- 失败/风险：若 `viewNum` 与标定映射错误，会表现为系统性偏移。
- 当前继承：`src/dataset.py` 使用 `positionID` 生成 `map_gt`，`src/config.py` 与 `src/calibration.py` 统一坐标常量。

## Phase 2: 几何投影与 warp 稳定性
- 问题：BEV warp 黑边、拉伸是否为实现错误。
- 尝试：`05_single_cam_bev_wildtrack_toolkit_style.py` 用 homography 与 projectPoints 双路径互证；`06_single_cam_feat_warp_to_bev.py` 验证 feature warp。
- 结论：黑边/拉伸在边界区域可为预期现象；feature-level warp 路径可用。
- 失败/风险：单位制（m/cm）或内参缩放不一致会导致 `valid_ratio` 降低和峰值偏移。
- 当前继承：`src/geometry.py` 的投影矩阵与 `warp_perspective_torch`，`src/train_main.py` 的 `valid_ratio` 过滤与单位推断。

## Phase 3: 单相机到多视角训练演进
- 问题：直接多视角训练易出现塌缩，难定位是数据、几何还是损失问题。
- 尝试：先用 `07_train_single_cam_occ.py` / `07_train_single_cam_occ_mvdet_style.py` 建立单相机可训练基线，再升级到 `08_*` 多视角版本。
- 结论：分阶段递进（单相机 -> 多视角）有助于快速定位问题来源；辅助头监督能提升稳定性。
- 失败/风险：aux GT 为空或解析错误时，aux loss 可能“看似正常但无学习”。
- 当前继承：`src/models.py` 保留 per-view `img_head`，`src/trainer.py` 保留 `aux_pos_mse` 指标。

## Phase 4: MVDet 风格对齐与工程化
- 问题：模型结构与损失是否贴近 MVDet 核心实现。
- 尝试：`08_train_multicam_mvdet_style_v2.py`、`_v3.py` 与 `final_train_multicam_mvdet_style.py` 逐步对齐损失和优化策略。
- 结论：`GaussianMSE`（pool + gaussian + mse）与 `SGD + OneCycleLR` 是关键收敛要素。
- 失败/风险：未对齐损失定义时，定位热图峰值不稳定。
- 当前继承：`src/loss.py` 的 `GaussianMSE`、`src/trainer.py` 的 `create_optimizer/create_scheduler`。

## Phase 5: Colab 自动化探索（已归档）
- 问题：如何降低手动操作成本，自动执行 Colab 训练。
- 尝试：`colab_automation_snapshot/` 中实现 Playwright 接管 Chrome、运行全部单元、错误收集与重试。
- 结论：自动化可提升重复执行效率，但易受页面结构变化影响，维护成本高。
- 失败/风险：选择器失效、登录状态、运行时连接变化导致不稳定。
- 当前替代：默认采用手动可控流程，仅保留自动化代码作知识资产。

## 关键排障手册（执行顺序）
1. 先看 `valid_ratio` 与视角过滤是否异常。
2. 再查 aux GT 是否非空（head/foot 热图是否有峰）。
3. 再看 `pos_mse` / `aux_pos_mse` 是否下降。
4. 若仍异常，回溯 `archive/legacy/training_prototypes/03~06` 的几何对照脚本。

## Traceability 索引
- 历史脚本：`archive/legacy/training_prototypes/`
- 自动化快照：`archive/legacy/colab_automation_snapshot/`
- 当前主链路：`src/train_main.py`, `src/models.py`, `src/dataset.py`, `src/trainer.py`, `src/loss.py`
