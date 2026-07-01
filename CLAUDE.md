# CLAUDE.md — BEV_Track-Predict 项目纪律

## 触发 CI/Colab 任务前的强制检查清单

每次触发 GitHub Actions 或 Colab 训练任务前，**必须按顺序完成以下检查**，不得跳过：

### 1. 验证前提条件

- **GPU 可用性**：首次使用某 GPU 型号时，先用 smoke test 验证（`echo "import torch; print(torch.cuda.get_device_name())" | colab exec`），确认 session 真正可用后再提交训练任务
- **数据源可达**：确认 Google Drive / OSS 等外部数据源未被限流。如果近期（1小时内）有多次下载失败，等待 cooldown 或切换数据源
- **代码正确性**：新增/修改的脚本，如果能本地验证（如可视化脚本用假数据），先本地跑通再提交

### 2. 分析前提条件

- **出错后禁止立刻重试**：任何 run 失败后，必须先查日志定位根因，确认修复后再触发下一个 run
- **区分基础设施问题 vs 代码问题**：连接超时、资源不可用、下载限流属于基础设施问题，重试代码不会解决；代码 bug 需要修改后才能重试
- **评估修复完整性**：一个修复是否真的解决了问题？是否只是把错误从一个地方移到了另一个地方？在触发前想清楚整条链路

### 3. 最小化测试

- **新功能先小规模验证**：新的训练配置先用 1 epoch + 少量 frames 跑通，确认流程无误后再全量训练
- **可视化先检查数据分布**：写可视化代码前，先打印数据的 min/max/mean/std，基于实际分布选择可视化方法，不凭直觉
- **模型质量匹配任务**：需要展示检测效果时，必须使用充分训练的模型（≥10 epochs），不用 1 epoch 的半成品

### 4. 执行纪律

- **关键产出就地导出**：checkpoint、指标、图片的 base64 导出必须在生成它们的同一个 `colab exec` 上下文中完成，不依赖后续步骤或 session 存活
- **禁止用 `|| true` 吞关键步骤的错误**：数据下载、checkpoint 下载、eval 执行等关键步骤失败必须报错，不得静默跳过
- **GPU 优先级**：优先 A100 > L4 > T4，但必须先验证可用性再使用。当前 Colab 免费账号仅 T4 可用
- **每次 run 触发前自问**：这个 run 和上一个失败的 run 相比，改变了什么？这个改变是否足以解决问题？

## 项目约定

- 训练配置：SGD, lr=0.05, OneCycleLR, grad_clip=1.0, concat fusion, resnet18 backbone
- BEV 网格：NB_HEIGHT=480 (i/rows), NB_WIDTH=1440 (j/cols), reduce_factor=4 → 120×360
- positionID 解码：`ix = pos % 480, iy = pos // 480`
- Gaussian kernel：MAP_SIGMA=2.2361 (variance=5.0), IMG_SIGMA=1.5811 (variance=2.5)
- 数据集：WildTrack 7 cameras, Google Drive folder ID: 1uBptJBbtMzVRQwSMRbQkIJp8-VVoBqUK
