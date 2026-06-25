# 当前工作状态 Handoff

> 写入时间：2026-06-25
> 项目路径：/Users/ster/Desktop/fusion/BEV_Track-Predict
> 语言要求：**全程使用中文回复**

---

## 正在进行的 GitHub Actions Run

**Run #28168973721**（当前正在运行，尚未完成）
- URL: https://github.com/sea-sky-web/BEV_Track-Predict/actions/runs/28168973721
- 配置：L4 GPU, epochs=10, max_frames=200, bev_pos_weight=10.0
- 关键修复：concat 融合 + SGD lr=0.1 + OneCycleLR + AMP + **gradient clipping (max_norm=10.0)**
- 新对话启动后，**第一件事是 `gh run view 28168973721 --json status,conclusion` 查看结果**

---

## 本轮完成的 4 项代码修复（已推送 main 分支）

| 修复 | 文件 | Commit |
|------|------|--------|
| concat 融合 + MVDetMapClassifier BEV head | src/models.py | 3bbfff5 |
| SGD lr=0.1 + OneCycleLR（backbone 不降 lr） | src/config.py, src/trainer.py | 3bbfff5 |
| ResNet-18 dilation 只对 conv1（去掉 conv2） | src/models.py | 3bbfff5 |
| NMS 半径 5→20 cells，max_preds 50→0 | src/evaluate_main.py | 3bbfff5 |
| 开启 AMP（--amp）防 concat OOM | scripts/colab_train.py | 8271cf4 |
| Gradient clipping max_norm=10.0 | src/trainer.py | b4b2500 |

当前 HEAD: b4b2500

---

## 失败历史（本轮）

| Run | 问题 | 修复 |
|-----|------|------|
| #28160557094 | T4 CUDA OOM（Epoch 4，concat 7视角太大） | 换 L4 + 开 AMP |
| #28163849009 | L4 + AMP，step 320 梯度爆炸（loss→5862→NaN） | 加 gradient clipping |
| **#28168973721** | **正在运行（等待结果）** | — |

---

## 项目背景

**目标**：WildTrack 多视角 BEV 行人检测，MODA ≥ 0.30

**本地 Eval 结果**（Run #46 旧模型，confidence_v2 + Adam，20 epoch）：
- MODA: -0.045（目标 ≥ 0.30）
- Recall: 3.4%（目标 ≥ 40%）
- 问题根因：融合架构差异 + 优化器策略差异

**当前方案**（MVDet 对齐）：
- concat 融合，BEV head = 3层 dilated conv（无BN，bias=False），输入 3586ch
- SGD lr=0.1 + OneCycleLR + momentum=0.5 + weight_decay=5e-4
- ResNet-18 dilation 只对 conv1
- NMS 半径 20 cells = 2.0m

**本地数据集**：`/Users/ster/Data/Wildtrack`（完整数据，可本地 eval）
**本地 Checkpoint**（旧模型，已不需要）：`/tmp/bev_ckpt/bev-checkpoint-run46/model_final.pth`

---

## 新对话的操作步骤

### 步骤 1：查看 run 结果
```bash
cd /Users/ster/Desktop/fusion/BEV_Track-Predict
gh run view 28168973721 --json status,conclusion
```

### 步骤 2a：如果成功，查看训练和 eval 结果
```bash
gh run view 28168973721 --log 2>/dev/null | grep -E "(\[Epoch [0-9]+\]|MODA|Precision|Recall|det_moda|exit code|non-finite|NaN)" | grep -v "^\s*\^" | head -30
```

### 步骤 2b：如果失败，查看错误
```bash
gh run view 28168973721 --log 2>/dev/null | grep -E "(error|ERROR|OOM|non-finite|NaN|exit)" | grep -v "^\s*\^" | head -20
```

### 步骤 3：如果 eval 成功，用本地数据做完整 eval
```bash
cd /Users/ster/Desktop/fusion/BEV_Track-Predict
# 先下载新的 checkpoint
gh run download 28168973721 -D /tmp/bev_ckpt_new

PYTHONPATH=src python3 src/evaluate_main.py \
  --data_root /Users/ster/Data/Wildtrack \
  --model_path /tmp/bev_ckpt_new/bev-checkpoint-run*/model_final.pth \
  --device cpu \
  --report_detection \
  --metrics_out /tmp/bev_eval_new.json \
  --views 0,1,2,3,4,5,6 \
  --max_frames 50 \
  --num_workers 0
```

---

## 已存文档

- `docs/mvdet_alignment_plan.md`：MVDet 对比分析和修复策略（详细）
- `docs/training_goals.md`：训练目标和验收标准
- `docs/eval_pipeline_analysis.md`：eval 管线修复历史

---

## 已知问题（待解决）

1. **eval 管线仍有 bug**：workflow 中的 eval 步骤用 `colab exec` 运行，但 Colab session 的 Python 版本可能无法访问负数阈值（parse_thresholds 的 bug 已在本地修复，但 Colab 上的代码版本需确认同步）
2. **Google Drive 限速**：wildtrack.zip 有时被限速，但最近几次训练 run 都成功下载了（限速已解除）
3. **bev_prediction.png 未生成**：visualize_prediction.py 依赖的数据路径问题，次要
