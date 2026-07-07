# CLAUDE.md — BEV_Track-Predict

> Claude Code 自动加载此文件。完整规则见 AGENTS.md。

## 文档地图

- `AGENTS.md` — AI 协作规则书（完整版）
- `docs/model_definition.md` — 模型架构约束（最高优先级）
- `docs/active_plan.md` — 当前迭代计划（每轮覆盖）
- `docs/LESSONS.md` — 经验沉淀（append-only，执行前必读）
- `docs/daily-log.md` — 每日实验日志（UTC+0）
- `docs/experiment_iteration_protocol.md` — 实验迭代规范
- `docs/dataset_contract.md` — 数据集契约
- `docs/training_goals.md` — 训练目标

## 核心纪律（摘要）

1. **独立判断**：禁止盲目赞同，用代码和数据支撑每个判断
2. **实证优先**：禁止"可能"，每个假设必须有证据链
3. **PR 流程**：所有变更通过 PR → CR → 合并
4. **训练审批**：触发 Colab run 前必须说明目的、差异、预期，用户批准后执行
5. **执行前读 LESSONS.md**：避免重蹈覆辙
