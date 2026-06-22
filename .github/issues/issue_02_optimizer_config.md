# [BUG][HIGH] Optimizer misconfiguration: SGD momentum=0.5 + OneCycleLR max_lr=0.1 causes unstable training

## Labels
`bug`, `high-priority`, `training`, `optimizer`

## Priority
**P1 — Causes divergence / oscillation even if pretrained weights are enabled**

---

## Problem Statement

The current optimizer configuration in `src/config.py` combines three settings that
are individually suboptimal and collectively harmful:

```python
# src/config.py
optimizer = "sgd"
lr        = 1e-3        # base learning rate
momentum  = 0.5         # abnormally low
weight_decay = 5e-4

scheduler = "onecycle"
max_lr    = 0.1         # 100× jump from base lr
```

This results in:

1. **Momentum too low (0.5)**: Standard SGD for vision uses momentum=0.9. At 0.5 the
   effective gradient smoothing is weak, causing high step-to-step variance and
   slow convergence.

2. **OneCycleLR max_lr=0.1 is 100× the base lr**: For a network starting from
   pretrained weights (the correct configuration post-issue #1), this level of
   peak learning rate will partially destroy the learned ImageNet features in the
   early epochs. MVDet uses Adam with lr=1e-4 throughout.

3. **SGD vs Adam on small datasets**: Adam is empirically more robust on small
   datasets with sparse positive supervision (sparse pedestrian heatmaps). SGD
   requires careful momentum and LR tuning that the current configuration gets wrong.

### Observed symptom

Training loss decreases in the first 1–2 epochs then plateaus or oscillates.
`pos_mse` (positive-region MSE) does not converge to near-zero, indicating the
network cannot fit even the training set.

---

## Proposed Fix

### Option A (Recommended): Switch to Adam — minimal change, well-validated

```python
# src/config.py
optimizer    = "adam"
lr           = 1e-4       # standard for fine-tuning pretrained backbone
weight_decay = 1e-4
scheduler    = "cosine"   # CosineAnnealingLR, T_max = total_epochs
```

Implementation in `src/trainer.py`:

```python
if cfg.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
elif cfg.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,    # must be 0.9
        weight_decay=cfg.weight_decay
    )
```

### Option B: Fix SGD configuration (if SGD is kept for reproducibility)

```python
# src/config.py
optimizer    = "sgd"
lr           = 1e-3
momentum     = 0.9         # fix: was 0.5
weight_decay = 5e-4
scheduler    = "onecycle"
max_lr       = 0.01        # fix: was 0.1, now 10× not 100×
pct_start    = 0.3         # warm-up for 30% of training
```

### Separate backbone and head learning rates

Regardless of optimizer choice, the backbone should use a lower LR than the
BEV head and fusion module:

```python
param_groups = [
    {"params": model.trunk.parameters(),      "lr": cfg.lr * 0.1},
    {"params": model.img_head.parameters(),   "lr": cfg.lr},
    {"params": model.fusion.parameters(),     "lr": cfg.lr},
    {"params": model.bev_head.parameters(),   "lr": cfg.lr},
]
optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
```

---

## Acceptance Criteria

- [ ] `src/config.py` updated with corrected optimizer defaults (Adam lr=1e-4 or SGD momentum=0.9)
- [ ] `src/trainer.py` supports `optimizer` and `scheduler` config fields cleanly
- [ ] Training loss curve shows monotonic decrease over 10 epochs (no oscillation)
- [ ] `pos_mse` drops by at least 50% from epoch 1 to epoch 10
- [ ] Experiment YAML updated with the validated optimizer config
- [ ] Old SGD+OneCycleLR config preserved as a named preset for ablation if needed

---

## References

- MVDet source code: Adam optimizer, lr=1e-4, no momentum
- "Bag of Tricks for Image Classification" (He et al., 2018): momentum=0.9 canonical
- PyTorch OneCycleLR docs: designed for max_lr ≈ 10× base_lr, not 100×
