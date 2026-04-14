# Colab 自动化 MVP（Playwright + 本地编排器）

这个目录提供了一个最小可运行的闭环自动化系统，对应你当前的训练流程：

1. 读取上一轮训练产物（`status.json`、`metrics.json`、`last_error.txt`）
2. 生成 AI patch 建议（当前是占位接口，后续可替换真实 LLM）
3. 按配置选择是否应用 patch
4. 自动 commit 并 push 到 GitHub
5. 使用 Playwright 打开 Colab notebook，连接 runtime，触发 Run All
6. 轮询结果文件直到成功或失败
7. 重复上述流程，直到成功或达到最大轮数

## 目录结构

```text
colab_automation/
  config/
    settings.py                # .env 加载与 AppConfig 定义
  launcher/
    playwright_launcher.py     # Colab 页面自动化控制
  monitor/
    result_monitor.py          # status/metrics/error 轮询与解析
  agent/
    patch_agent.py             # Patch 接口 + 占位实现
  gitops/
    git_client.py              # Git 提交与推送封装
  orchestrator/
    main.py                    # 主闭环编排控制器
  docs/
    notebook_result_protocol.md# Notebook 输出协议
  .env.example                 # 配置模板
  requirements.txt             # 本 MVP 额外依赖（Playwright）
  init_colab_login.py          # 初始化持久化登录态脚本
  run_orchestrator.py          # 主入口脚本
```

## 从零启动

1. 安装依赖：

```bash
pip install -r colab_automation/requirements.txt
python -m playwright install chromium
```

2. 准备配置：

```bash
cp colab_automation/.env.example colab_automation/.env
```

然后编辑 `colab_automation/.env`，重点配置：

- `COLAB_NOTEBOOK_URL`：固定 Colab notebook 地址
- `REPO_PATH`：本地仓库路径
- `RESULT_DIR`：训练结果文件同步目录
- `GIT_BRANCH`：推送目标分支

3. 选择浏览器控制模式（两选一）：

- 模式 A：启动新实例（默认）
  1. 执行登录初始化脚本：
     ```bash
     python -m colab_automation.init_colab_login
     ```
  2. 脚本会打开带持久化 profile 的浏览器窗口。手动登录后回车保存。

- 模式 B：附着到“已登录”的 Chrome（你当前需求）
  1. 先手动启动一个带 CDP 端口的 Chrome（Windows 示例）：
     ```powershell
     "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
     ```
  2. 在这个 Chrome 里确认你已登录 Google 且可打开 Colab。
  3. 配置 `.env`：
     - `PLAYWRIGHT_ATTACH_EXISTING_CHROME=true`
     - `PLAYWRIGHT_CDP_URL=http://127.0.0.1:9222`
     - `PLAYWRIGHT_ATTACH_NEW_TAB=true`（建议保持 true，避免污染你当前标签页）

4. 启动自动闭环：

```bash
python -m colab_automation.run_orchestrator --log-level INFO
```

## 关键配置项（`.env`）

- `COLAB_NOTEBOOK_URL`：Colab notebook URL
- `REPO_PATH`：本地 Git 仓库根目录
- `GIT_BRANCH`：推送分支
- `RESULT_DIR`：包含结果文件的目录
- `POLL_INTERVAL_SECONDS`：结果轮询间隔
- `ROUND_TIMEOUT_SECONDS`：单轮最大等待时间
- `MAX_ROUNDS`：最大循环轮数
- `PLAYWRIGHT_PROFILE_DIR`：Playwright 持久化浏览器 profile 目录（模式 A 使用）
- `PLAYWRIGHT_ATTACH_EXISTING_CHROME`：是否附着已有 Chrome
- `PLAYWRIGHT_CDP_URL`：附着模式使用的 CDP 地址
- `PLAYWRIGHT_ATTACH_NEW_TAB`：附着时是否新开标签页
- `MODEL_CONFIG_PATH`：可选，供 patch 模块读取的配置文件路径
- `AUTO_APPLY_PATCH`：`true/false`，是否自动应用生成的 patch

## Notebook 最小改造要求

Notebook 需要稳定输出以下四个文件：

- `status.json`（必须包含 `status`，建议包含 `round_id`）
- `metrics.json`
- `train.log`
- `last_error.txt`

协议和示例代码见：

- `colab_automation/docs/notebook_result_protocol.md`

## 已知脆弱点（后续需要维护）

1. Colab UI selector 变化：
   - “Connect”按钮、“Run all”菜单文案可能因语言或 A/B 实验变化而失效。
   - 调整文件：`launcher/playwright_launcher.py`。
2. Runtime 连接策略变化：
   - 如果出现额外的人机校验或账号风控弹窗，自动化可能中断。
3. 文件同步延迟：
   - 如果 `RESULT_DIR` 依赖 Google Drive 同步，本地可见时间可能延迟几十秒。
4. 状态文件新鲜度：
   - 若 notebook 未写入终态 `status.json`，监控器会超时。
5. 占位 patch 能力有限：
   - 当前 `HeuristicPatchAgent` 只做简单规则，不会自动修复复杂模型问题。
6. 附着模式前置条件：
   - Chrome 必须带 `--remote-debugging-port` 启动，普通启动的 Chrome 无法被附着。

## MVP 后续增强方向

1. 将 `HeuristicPatchAgent` 替换为真实 LLM：
   - 保持 `PatchRequest -> PatchPlan` 接口不变。
2. 增加分轮产物隔离：
   - 例如输出到 `round_{id}/status.json`。
3. 增加 PR 流程：
   - 每轮推送到独立分支并自动创建 Draft PR。
4. 增加健康检查：
   - 检测 Colab 卡死、kernel 无响应并自动重启流程。
5. 增加实验调度：
   - 支持多配置任务队列与优先级。

