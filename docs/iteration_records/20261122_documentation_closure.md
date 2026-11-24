# 2026-11-22 文档闭环完善

- 时区：Asia/Shanghai。
- 类型：文档；状态：已结束（文档检查通过，未提交）。
- 授权：用户要求仅完善现有文档体系，不进行代码或模型改动。
- 文档起草于 ai-iteration-aux-alpha-control（379ec8e9338104e0ae286efd2158ac798c9f9da1），随后工作目录切换至 docs/paper-readiness-audit（d372119803191c749fb047ac8ad7eceae8987152）。
- 上一轮依据：本会话的仓库审计、远端运行核对和用户明确的文档范围要求；原始运行证据见下表。

## 依据与计划

现有五层记录足以使用，但缺统一入口、简洁模板及跨会话维护要求。本轮补齐这三项，并把已知问题登记到可追溯记录中。保留历史目录与现有格式；不引入平台或自动化代码。

验证标准：入口能链接到当前行动、模板、日志和实验规范；模板覆盖训练前依据与训练后证据；只修改 Markdown；不改历史 ai_runs，不触发训练。

## 执行与结果

本轮修改和验证由 Git 工作区差异及本记录说明。本轮没有训练、模型评估或新指标，因此不在关键事件日志中创建条目。文件式闭环通过人工维护执行，不代表自动强制机制已经存在。

## 历史证据与问题登记

以下为本会话前序审计的整理，不是本轮新实验或远端重查。截止日期 2026-11-22；原始归档缺口继续保留。

| 问题 | 依据 | 状态/限制 | 后续验收 |
|---|---|---|---|
| 产物保全 | [Run 101](https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/33735428270)、[102](https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/33747560484)、[103](https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/33755384172)、[104](https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/33829519054) | 未确认永久备份；前序查得 12 月 2–3 日到期；Run 102 下载审批额度受阻 | 持久保存并回读核验，登记文件校验值 |
| 分支与计划不一致 | [审计版本](https://github.com/sea-sky-web/BEV_Track-Predict/tree/d372119803191c749fb047ac8ad7eceae8987152) | 本地仍在旧分支；审计分支仅为候选开发基线 | 后续整合保留双方记录，明确真实执行 SHA |
| 超时仍显示工作流成功 | Run 101/103，前序原始日志核对 | 未修复；本轮不改代码 | 后续单独修复和验证失败传播 |
| 对照预算不一致 | Run 102/104：test MODA 0.8035714 / 0.8445378；训练 5 / 9 epoch | 仅单 seed、同时改变骨干/融合；不能归因于单项创新 | 相同预算与协议重跑，另行授权 |
| 测试与可视化失败 | [Smoke](https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/33830798441) 1 failed / 30 passed；Run 104 可视化模型不匹配 | 未修复；数值评估与画图状态分开 | 单独定位并验证 |
| M2 结论有效性 | [指标源码](https://github.com/sea-sky-web/BEV_Track-Predict/blob/d372119803191c749fb047ac8ad7eceae8987152/src/temporal/tracking_metrics.py)、[流水线](https://github.com/sea-sky-web/BEV_Track-Predict/blob/d372119803191c749fb047ac8ad7eceae8987152/scripts/run_m2_pipeline.py) | 身份指标、L2/L3 数据流、未来目标、尺度和首点未来信息问题未修复 | 专项修复后重评，历史结论待验证 |

历史综合结果参考 [M1 归档](https://github.com/sea-sky-web/BEV_Track-Predict/blob/cc22a08de727c8a86d0e623c5a97c4f96455046d/docs/module1_experiment_data_archive.md) 与 [审计报告](https://github.com/sea-sky-web/BEV_Track-Predict/blob/d372119803191c749fb047ac8ad7eceae8987152/docs/paper_readiness_audit.md)。旧 MODA 0.8950 涉及测试集选参等限制，不作为最终成绩。

## 收尾

- 本轮结论：文档入口、当前行动、迭代模板和关键事件日志的职责已明确；只修改文档，不声称模型改善。
- 后续唯一主要动作：核实并保存已有实验产物；本轮未执行。
- Git 状态：文档迁移至审计分支工作区，未提交、未推送；旧分支修改另存 Git stash 和临时备份。

## 分支核对快照（2026-11-22）

审计分支 `d372119` 包含 `main` 的全部提交与后续审计修正，是候选开发基线；当前工作区已切换至审计分支。其独有几何补丁基于旧坐标布局，不能直接套用。分支状态为本次核对快照，后续操作前重新查询。
