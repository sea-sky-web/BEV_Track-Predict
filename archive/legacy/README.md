# Legacy Archive 索引

本目录用于保留历史探索，不作为默认执行入口。

## 目录结构
- `training_prototypes/`：早期单文件/阶段式训练与几何验证脚本。
- `colab_automation_snapshot/`：已删除的 Colab 浏览器自动化链路快照（从 `HEAD` 恢复）。

## 归档文件价值映射

### training_prototypes
- `02_draw_bboxes.py`
  - 价值：验证 Wildtrack `views`/bbox 读取闭环。
  - 当前替代：`src/dataset.py` 的 `views` 标签构造。
- `03_geom_closure_validate.py`
  - 价值：`positionID -> 世界坐标 -> 投影` 几何闭环诊断。
  - 当前替代：`src/calibration.py` + `src/geometry.py` + `src/train_main.py` 的 `valid_ratio` 打印。
- `04_find_cam_mapping.py`
  - 价值：相机编号与标定名称映射的鲁棒排查。
  - 当前替代：`src/config.py` 的 `CAM_NAMES` 与 `CalibrationLoader`。
- `05_single_cam_bev_wildtrack_toolkit_style.py`
  - 价值：homography 与 projectPoints 双路径对照验证。
  - 当前替代：`src/geometry.py` 的投影矩阵链路。
- `06_single_cam_feat_warp_to_bev.py`
  - 价值：从 RGB warp 过渡到 feature warp 的可行性验证。
  - 当前替代：`src/models.py` 中 `warp_perspective_torch` 调用。
- `07_train_single_cam_occ.py`
  - 价值：单相机快速训练原型与 OOM/速度经验。
  - 当前替代：多视角主链路 `src/train_main.py`。
- `07_train_single_cam_occ_mvdet_style.py`
  - 价值：MVDet 风格损失和单相机对齐实验。
  - 当前替代：`src/loss.py` 的 `GaussianMSE` 与 `src/trainer.py`。
- `08_train_multicam_mvdet_style.py` / `_v2.py` / `_v3.py`
  - 价值：多视角融合、投影稳定性、辅助头训练策略演进轨迹。
  - 当前替代：模块化主链路 `src/*`。
- `final_train_multicam_mvdet_style.py`
  - 价值：单文件收敛版实现，便于追溯完整实验配置。
  - 当前替代：`src/train_main.py` + `src/models.py` + `src/trainer.py`。

### colab_automation_snapshot
- `STARTUP_GUIDE.md`
  - 价值：远程调试 Chrome + Colab 自动化操作步骤。
  - 当前替代：保留文档知识，不作为默认流程。
- `run_colab_training.py`
  - 价值：Colab 内“一键 clone + install + train/eval”的早期入口。
  - 当前替代：`colab.ipynb` 直接调用 `src/train_main.py` / `src/evaluate_main.py`。
- `run_automated_training.py`
  - 价值：自动 push + 自动执行 notebook + 错误循环重试。
  - 当前替代：手动可控流程 + `errors/` 报告机制。
- `test_browser_control.py` / `debug_colab.py`
  - 价值：页面结构变更与选择器失效时的排障工具。
  - 当前替代：仅在需要恢复自动化时参考。
- `colab_automation/`
  - 价值：Playwright 启动器、执行器、错误收集器完整实现。
  - 当前替代：归档保留，不进入当前训练主路径依赖。
