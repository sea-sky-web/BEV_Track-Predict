# Plan: 细粒度阈值 + NMS 半径扫描

## Context
MODA 0.793 vs 目标 0.882，差距 0.089（FP=106, FN=91）。当前评估使用固定 NMS 半径 5.0 和 13 个阈值点。通过更细粒度的参数扫描，可能找到更优的操作点，无需重训。

## 修改内容

### 1. `scripts/colab_train.py` — 细化阈值扫描
当前 13 个点，在最优区间 0.25-0.55 加密到 0.025 步长：
```
--det_thresholds=-0.50,-0.25,-0.10,0.00,0.05,0.10,0.15,0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50,0.55,0.60
```

### 2. `src/evaluate_main.py` — 支持多 NMS 半径扫描
添加 `--det_min_distances` 参数（逗号分隔），遍历每个 NMS 半径独立运行完整的阈值扫描，输出每组 (nms_radius, threshold) 的 MODA，最终选择全局最优组合。

- 新增 `parse_min_distances()` 函数
- `evaluate_detection()` 无需修改（已接受 min_distance 参数）
- 在 `main()` 中循环调用，收集所有组合的结果
- 保留 `--det_min_distance` 单值参数的向后兼容

### 3. `scripts/colab_train.py` — 传入多 NMS 半径
```
--det_min_distances=3.0,4.0,5.0,6.0,7.0,8.0
```

## 关键文件
- `src/evaluate_main.py` — 评估主脚本
- `scripts/colab_train.py` — Colab 训练/评估入口

## 验证
- 触发一次 train+eval run（需用户批准）
- 从日志中提取所有 (nms_radius, threshold) 组合的 MODA
- 选择全局最优组合
