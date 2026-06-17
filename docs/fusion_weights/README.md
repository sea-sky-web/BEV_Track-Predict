# Fusion Weight Artifacts

This directory is reserved for `confidence_v2` per-view weight maps generated from a real checkpoint:

```bash
python scripts/visualize_fusion_weights.py --data_root wildtrack --model_path outputs/train_multicam_mvdet_style_v3/model_final.pth
```

Do not claim attention-weight behavior without checkpoint-backed images.
