# scripts 目录说明

本目录仅保留**当前活动脚本**。

## 活动入口
- `verify_modules.py`：对 `src/` 模块进行导入与接口完整性检查。
- `calibration.py`：标定相关的独立工具模块（保留为可复用脚本）。
- `train_main.py`：兼容入口，路由到 `src/train_main.py`。
- `run_colab_exp.py`：Colab 实验 launcher，写出 metrics/config 记录。
- `commit_ai_runs.py`：整理并提交 Colab 运行结果到 `ai_runs/`。
- `visualize_projection.py`：基于真实 WildTrack 数据生成投影覆盖 overlay。
- `visualize_fusion_weights.py`：基于真实 checkpoint 导出 `confidence_v2` per-view 权重热图。

## 已归档历史探索
以下历史训练原型已迁移到：
- `archive/legacy/training_prototypes/`

迁移范围：
- `02_draw_bboxes.py`
- `03_geom_closure_validate.py`
- `04_find_cam_mapping.py`
- `05_single_cam_bev_wildtrack_toolkit_style.py`
- `06_single_cam_feat_warp_to_bev.py`
- `07_train_single_cam_occ.py`
- `07_train_single_cam_occ_mvdet_style.py`
- `08_train_multicam_mvdet_style.py`
- `08_train_multicam_mvdet_style_v2.py`
- `08_train_multicam_mvdet_style_v3.py`
- `final_train_multicam_mvdet_style.py`

## 当前主训练链路
请使用：
- `src/train_main.py`（训练）
- `src/evaluate_main.py`（评估）

当前默认链路为 ResNet-18 pretrained backbone、`confidence_v2` fusion、Adam + cosine、全 7 视角、全帧训练。
