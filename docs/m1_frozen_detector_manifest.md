# Module 1 Frozen Detector Manifest

> 创建日期：2026-07-15
> 状态：checkpoint 待下载验证，指标来自 GA 训练日志

---

## 1. Checkpoint 来源

| 项目 | 值 |
|------|-----|
| GA Run ID | 29345199882 |
| GA Run Number | 95 |
| Artifact Name | bev-checkpoint-run95 |
| Artifact ID | 8319492725 |
| Artifact Size | 21,714,456 bytes |
| Commit SHA | 014d33258829eb61f827f2c6eb5048e1cfa4318d |
| Tag | v0.2.1-moda8950 |
| Training Date | 2026-07-14 |
| Previous Best | Run 29332987206 (cv2, MODA 0.8918) |

## 2. 冻结配置

```yaml
backbone: mobilenet_v2    # truncated features[0:14], gradient checkpointing
fusion_mode: geo_confidence_v1
feat_ch: 512
views: 0,1,2,3,4,5,6      # 7 views
optimizer: sgd
lr_init: 0.1
max_lr: 0.1
momentum: 0.5
weight_decay: 0.0005
scheduler: onecycle
epochs: 10
batch: 1
max_frames: 360            # train split
pretrained: true
augment: false
loss_type: mse
offset_weight: 0.0
bev_pos_weight: 1.0
```

## 3. 冻结评估配置

```yaml
frame_start: 360           # test split (frames 360-399)
max_frames: 40
det_best_threshold: 0.375
det_best_nms_radius: 5.0
det_dist_thr: 3.0          # MODA matching distance (BEV cells)
```

## 4. 冻结指标（来自训练 run 日志）

| 指标 | 值 |
|------|-----|
| MODA | 0.8950 |
| MODP | 0.7778 |
| Precision | 0.9301 |
| Recall | 0.9223 |
| F1 | 0.9262 |
| TP | 898 |
| FP | 46 |
| FN | 54 |
| Best Threshold | 0.375 |
| Best NMS Radius | 5.0 |

## 5. 训练曲线

```
Epoch 0: loss=0.011759 bev=0.010168 img=0.001591 snr=0.213
Epoch 1: loss=0.006300 bev=0.005365 img=0.000935 snr=0.280
Epoch 2: loss=0.004836 bev=0.003998 img=0.000838 snr=0.304
Epoch 3: loss=0.003860 bev=0.003048 img=0.000812 snr=0.318
Epoch 4: loss=0.003492 bev=0.002690 img=0.000803 snr=0.328
Epoch 5: loss=0.003307 bev=0.002508 img=0.000799 snr=0.334
Epoch 6: loss=0.003123 bev=0.002326 img=0.000797 snr=0.340
Epoch 7: loss=0.002955 bev=0.002159 img=0.000795 snr=0.346
Epoch 8: loss=0.002768 bev=0.001974 img=0.000794 snr=0.352
Epoch 9: loss=0.002633 bev=0.001839 img=0.000794 snr=0.357
```

## 6. 模型参数量

| 组件 | 参数量 |
|------|:------:|
| Backbone (MobileNet-V2 truncated) | 0.6M |
| ImgHead | 0.6M |
| Attention Fusion (ConcatAttentionFusion) | 1.84M |
| BEV Head (BEVHeadDilated) | 2.4M |
| Offset Head | 0.3M |
| **Total** | **5.7M** |

## 7. 待完成验证

- [ ] 下载 checkpoint 到本地，计算 SHA256
- [ ] 在 Colab 上用冻结配置复核评估指标
- [ ] 导出 test split 的检测点 JSONL（frame_index, world_x_m, world_y_m, score）
- [ ] 将 checkpoint 和 manifest 归档到 outputs/frozen_detector/

## 8. Module 2 使用约束

Module 2 的所有时序实验必须：
1. 使用本 manifest 中记录的 checkpoint，不得更换
2. 使用冻结的 threshold=0.425 和 NMS=6.0 提取检测点
3. 记录 detector manifest 版本号和 checkpoint SHA256
4. 分层报告：GT → GT association → full tracker 三级评估
