# AI Context — 本次对话全部错误记录与反思

## 实验信息

| 字段 | 值 |
|------|-----|
| 对话日期 | 2026-06-30 ~ 2026-07-01 |
| 起始 commit | c423e2b (三个根因 bug 修复) |
| 结束 commit | 07dc688 |
| 涉及 GA Runs | 28364780247, 28364788716, 28415093486, 28417189000, 28417512342, 28417638623, 28418202813, 28425305574, 28437526248, 28489265047, 28494973746 |

## 本次对话成果

- 逐层对比验证 ALL PASS（投影矩阵、coord map、GT 热图、Gaussian kernel）
- MODA 从 0.0000 首次突破至 **0.5723**（全量 1800 frames × 10 epochs）
- Precision 0.9491, MODP 0.8298（已超 MVDet 论文的 0.726）

---

## 错误 1：盲目使用 A100 GPU 导致连续失败

### 现象
两次 A100 run（28417189000, 28417512342）都在 session 创建成功后立刻断开连接：
- Run 1: Session READY → 37 秒后 HTTPSConnectionPool read timeout
- Run 2: Session READY → 12 秒后 RuntimeError: Connection was lost

### 错误原因
没有验证当前 Colab 账号是否支持 A100。免费版 Colab 不提供 A100，`colab new --gpu A100` 
表面返回 READY 但实际 kernel 不可用，colab-cli 没有正确报错。

### 犯错根因（反思）
用户提出"优先使用 A100"的需求后，**没有先做可行性验证**就直接触发了训练任务。
应该先用一个简单的 smoke test（如 `echo "print('hello')" | colab exec`）确认 A100 session
是否真正可用，而不是用耗时 2-3 小时的全量训练来"试错"。

### 修正
降级到 T4 成功运行。但浪费了 2 次 run 和约 30 分钟的排查时间。

### 教训
**对基础设施变更（GPU 型号、资源配置）应先做最小化验证，再投入完整任务。**

---

## 错误 2：首次全量训练因 Google Drive 限流失败

### 现象
GA run 28417638623（T4, 1800 frames）：gdown 下载 wildtrack.zip 失败，
报 "Cannot retrieve the public link of the file"。训练直接跳过。

### 错误原因
短时间内多次触发 GA run，每次都重新从 Google Drive 下载 ~2GB 的 wildtrack.zip，
触发了 Google Drive 的公开下载频率限制。脚本中 gdown 失败后直接 `sys.exit(1)`，
没有任何 fallback 机制。

### 犯错根因（反思）
1. 没有意识到频繁触发 run 会累积 Drive 下载次数
2. 数据下载脚本没有设计 fallback（如检查本地缓存、使用备用下载源）
3. 取消上一个 run 再立刻触发新 run，中间没有考虑 cooldown

### 修正
给 `colab_train.py` 添加了 fallback 路径逻辑：gdown 失败时检查
`/content/wildtrack`, `/content/BEV_Track-Predict/wildtrack` 等缓存路径。
但由于每次 `colab new` 创建全新环境，fallback 路径在新 session 上也是空的，
所以这个修复只在 session 复用场景有效。

### 根本解决（未做）
应该将 wildtrack 数据集上传到更稳定的存储（如 GitHub Release、OSS），
或在 Colab 的 Google Drive 挂载中持久化。

### 教训
**对外部依赖（第三方存储下载）应有降级策略，且要考虑频率限制。**

---

## 错误 3：Artifact 上传全部失败，可视化图片无法获取

### 现象
多个 run（28418202813, 28425305574, 28489265047）的 `actions/upload-artifact@v4` 
都报 "No files were found with the provided path"。checkpoint、eval 结果、
可视化图片全部丢失。

### 错误原因
Workflow 设计缺陷：训练、下载、eval、可视化被拆成多个独立步骤，
每个步骤用单独的 `colab download` 从 Colab 拉取文件。但 Colab session 
在长时间训练后变得不稳定（或已断开），后续的 `colab download` 静默失败
（被 `|| true` 吞掉），GA runner 上根本没有文件，artifact 上传自然失败。

### 犯错根因（反思）
1. **过度依赖 session 存活**：假设 Colab session 在训练完成后仍然可用
2. **错误使用 `|| true`**：让所有下载失败都静默通过，没有报警或 fallback
3. **没有端到端验证**：加了 `colab download` 但从未验证它是否真的能在训练后工作

### 修正（迭代 3 次）
1. 第一次修：在 eval+viz 的 `colab exec` 里加 base64 导出 → 但 eval 步骤是独立的，
   session 已断开时这个步骤也跑不了
2. 第二次修：在 GA runner 端加 "Extract base64 from log" 步骤解码日志中的 base64 → 
   但 `gh run view` 在 run 未完成时拿不到完整日志，解码失败
3. 第三次修（最终）：把 base64 导出放进 `colab_train.py` 本身，训练完立刻输出，
   不依赖任何后续步骤或 session 存活

### 教训
**关键产出（checkpoint、指标、图片）的导出应该和生成它们的过程在同一个执行上下文中完成。
不要假设远程 session 会一直存活。**

---

## 错误 4：可视化效果全蓝/无对比度

### 现象
用户检查 bev_prediction.png 后反馈"上图 GT 一片蓝，下图什么也看不出来"。

### 错误原因
1. **GT 热图**：在 120×360 网格上只有 ~38 个 Gaussian 中心点（每个只有几个像素），
   99.9% 的区域值为 0。JET colormap 把 0 映射为蓝色，极小的亮点肉眼不可见。
2. **Prediction 热图**：模型输出 logit 范围 [-0.24, 1.07]，sigmoid 后所有像素
   都压缩到 [0.47, 0.75]。用 sigmoid 值 × 255 做颜色映射，整张图只用了
   JET 色带的一小段，几乎没有对比度。

### 犯错根因（反思）
1. 写可视化脚本时**没有检查数据的实际数值分布**就选择了可视化方法
2. 直接用 sigmoid（它的作用是映射到概率，不是用于可视化）做热图
3. 没有在本地生成小图验证就直接提交到 CI 运行
4. 第一次修复（归一化 sigmoid）仍然不对，因为 sigmoid 本身就把所有值压到 ~0.5

### 修正
1. GT：二值膨胀（7×7 椭圆核）+ 绿色圆圈标注行人位置
2. Prediction：用 raw logit（裁掉负值，按最大值归一化到 [0,255]）
3. 整体放大 3 倍（120×360 → 360×1080）
4. Overlay 图：绿圈 = GT，红圈 = 检测峰值（local maxima of raw logit > 0.3）

### 教训
**可视化代码必须基于数据的实际分布来设计，不能凭直觉选方法。
写完后应先在本地小规模验证再提交。**

---

## 错误 5：用 1 epoch 模型做可视化验证

### 现象
为了"快速拿到可视化图片"，触发了 1 epoch / 200 frames 的短训练（28437526248）。
结果 sigmoid mean=0.51, sigmoid>0.3 占 100% 像素 — 模型完全没学到区分前背景。

### 错误原因
1 epoch 的模型几乎等于随机初始化，输出的 logit 全在 0 附近，
sigmoid 后全部 ≈ 0.5，不可能产生有意义的可视化。

### 犯错根因（反思）
急于展示结果，没有思考"1 epoch 的模型是否足够产生可判别的输出"。
之前全量训练的 MODA=0.57 说明 10 epoch 模型可以区分行人，
但 1 epoch 模型的 SNR 只有 ~0.44 且 sigmoid 输出近乎均匀，
根本不适合做可视化验证。

### 修正
重新用 10 epoch / 1800 frames 训练生成可视化。

### 教训
**验证任务也需要匹配足够的模型质量。"快速验证"不等于"用最差的模型验证"。**

---

## 错误 6：反复触发无效 GA Run 浪费资源

### 现象
本次对话共触发 11 个 GA run，其中：
- 2 个 A100 连接失败（浪费）
- 1 个 T4 gdown 限流失败（浪费）
- 2 个 artifact 导出失败（部分浪费 — 训练跑了但结果拿不到）
- 1 个 1 epoch 无意义可视化（浪费）
- 实际有效 run 仅 3-4 个

### 犯错根因（反思）
1. **没有在触发前做充分的预检查**（GPU 可用性、Drive 限流、artifact 流程）
2. **出错后立刻重试而不是先分析根因**（A100 失败 → 立刻重试 A100，而不是先查原因）
3. **修复不彻底就重新触发**（base64 导出改了 3 次才放对位置）

### 教训
**每次触发长时间 CI 任务前，应该：
1. 确认基础设施可用（GPU、数据源、网络）
2. 确认产出链路完整（训练 → 导出 → 上传）
3. 本地能验证的先本地验证（如可视化脚本可以用假数据本地测试）**

---

## 总结与改进计划

### 核心问题
本次对话的大部分错误源于**急于执行、疏于验证**的模式：
- 用户提需求 → 立刻触发 run → 失败 → 修一点 → 再触发 → 再失败
- 正确的模式应该是：分析 → 验证前提条件 → 最小化测试 → 确认可行 → 完整执行

### 改进项
1. **GPU 选择**：先 smoke test 再决定 GPU 型号
2. **数据下载**：将 wildtrack 迁移到更稳定的存储，或在 Colab Drive 挂载中持久化
3. **文件导出**：所有关键产出在生成的同一 exec 上下文中导出（base64 to stdout）
4. **可视化**：基于实际数值分布设计，本地先验证
5. **CI 触发纪律**：每次触发前检查 checklist，避免无效 run
