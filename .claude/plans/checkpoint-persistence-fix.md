# Plan: 修复 Checkpoint 存储可靠性（高风险 — CI/CD 配置变更）

## 需求细化

**问题**：训练完成后，`colab download` 经常失败，导致 checkpoint/eval 结果丢失，需要重新花费 GPU 时间训练。

**已验证的失败原因**（非推测）：
1. PR #74 的 Google Drive 持久化方案 `google.colab.drive.mount()` 从未生效 —
   实际日志证据：`[WARN] Google Drive 保存失败: mount failed`（run 28845973141, 07:34:09）
2. colab-cli 官方文档（`colab skill`）明确写明：
   `colab drivemount` — **"interactive; not agent-runnable"**
   `colab auth` — **"interactive; not agent-runnable"**
   → Drive 持久化路径在无人值守场景下**根本不可能生效**，不是实现问题，是能力边界问题
3. 根因：`colab download` 是在 `colab exec` **返回之后**作为独立 CLI 调用执行的
   （`.github/workflows/colab-train.yml` "Run training and download results" 步骤）。
   Colab session 在脚本结束后的这个窗口期经常变得无响应
   （历史日志中出现 Jupyter kernel message queue timeout stack trace）
4. Base64 stdout 导出已被用户明确否决（乱码问题，见 daily-log.md 07-01 记录）

**目标**：即使 session 在训练脚本结束后立即失联，也不丢失（或最多丢失几分钟的）训练成果。

**边界**：
- 范围内：`.github/workflows/colab-train.yml`（后台轮询下载）、`scripts/colab_train.py`（移除失效的 Drive 步骤）
- 范围外：不引入 Google Drive API / service account（增加新的认证复杂度和 secret 管理）；不用 base64（已被否决）；不改动训练循环或 checkpoint 格式；不影响 `checkpoint_run` eval-only 流程

## 风险等级：高

按 `docs/rules/risk-levels.md`，CI/CD 配置变更明确列为高风险。本方案包含备选对比、回退方案。

## 方案设计

### 核心思路：训练期间周期性下载，而非训练结束后才下载

利用已验证的事实：`model_final.pth` 每个 epoch 结束都会更新（`src/train_main.py:421`）。
把 `colab exec` 放到后台运行，同时启动一个循环，每 3 分钟尝试下载最新的
`model_final.pth` / `eval_results.json`（失败则忽略，继续训练）。
这样即使 session 在脚本结束瞬间失联，我们已经拿到了几分钟前的最新权重。

### 具体改动

**`.github/workflows/colab-train.yml`** — "Run training and download results" 步骤：

```bash
LOGFILE=train_exec.log
{
  echo "import sys; sys.argv = [...]"
  cat scripts/colab_train.py
} | colab exec -s bev-train --timeout 7200 > "$LOGFILE" 2>&1 &
EXEC_PID=$!

tail -f "$LOGFILE" &
TAIL_PID=$!

REMOTE="/content/BEV_Track-Predict/outputs"
while kill -0 $EXEC_PID 2>/dev/null; do
  sleep 180
  colab download -s bev-train "$REMOTE/train_multicam_mvdet_style_v3/model_final.pth" ./model_final.pth || true
  colab download -s bev-train "$REMOTE/eval_results.json" ./eval_results.json || true
done

wait $EXEC_PID
EXEC_RET=$?
kill $TAIL_PID 2>/dev/null || true

# 训练结束后的最终尝试（可能成功，也可能是本次唯一机会失败但已有周期性备份）
for f in eval_results.json train_multicam_mvdet_style_v3/model_final.pth \
         train_multicam_mvdet_style_v3/metrics.csv \
         visualization/bev_prediction.png visualization/bev_overlay.png; do
  colab download -s bev-train "$REMOTE/$f" "./$(basename "$f")" || true
done

exit $EXEC_RET
```

**`scripts/colab_train.py`** — 移除第 7 步失效的 Drive 挂载代码（`drive.mount()` 调用及其 try/except 包裹的整段逻辑），恢复为 6 步流程。这段代码 100% 会失败，保留它只会每次浪费约 2 分钟并打印误导性日志。

### 备选方案对比

| 方案 | 说明 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **A（推荐）：训练期间周期性下载** | 见上 | 复用已验证可用的 `colab download` 原语；无新依赖；最多损失几分钟训练 | `colab download` 与 `colab exec` 并发调用的可靠性未经实测，需 smoke test 验证 | **采用** |
| B：按 epoch 拆分为多次 `colab exec` | 每个 epoch 单独 exec，之间下载 | session 存活窗口更可控 | 每次 exec 都要重新 clone/pip install/查数据，10 epoch 累积开销大；训练代码不支持 checkpoint-resume，需额外改动 | 不采用（改动面过大） |
| C：仅移除失效 Drive 代码，不加新机制 | 什么都不修 | 零风险 | 不解决问题 | 不采用 |
| D：Google Drive API + service account | 绕过 colab-cli，直接用已有 OAuth token（含 drive.file scope）调 Drive API 上传 | 理论可行 | 需要把 refresh token 注入到不受信的 Colab exec 上下文；`drive.file` scope 权限范围存疑；新增认证代码路径 | 不采用（复杂度/风险不匹配收益，A 已足够） |

### 回退方案

纯 workflow YAML 变更，`scripts/colab_train.py` 改动只是删除死代码。如果新方案表现异常（如后台轮询导致日志顺序混乱、`colab download` 并发调用报错阻塞训练），执行 `git revert` 恢复上一版本即可，不影响任何其他系统组件。

## 验证计划

1. YAML 语法校验：`python3 -c "import yaml; yaml.safe_load(open('.github/workflows/colab-train.yml'))"`
2. 触发一次**低成本**真实 Colab run 验证（需用户批准）：epochs=1, max_frames=20, T4 GPU，验证：
   - 后台执行不破坏日志实时可见性
   - 训练期间至少有一次周期性下载成功
   - 即使最终下载失败，已通过周期性下载拿到产出物
3. 对比：本次改动只影响 workflow 执行方式，不影响训练/评估逻辑，因此结果指标（MODA 等）预期与之前一致，重点验证"文件是否到手"而非"指标是否变化"
