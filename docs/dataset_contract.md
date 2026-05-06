# Dataset Contract

## 1. Purpose

This document defines the dataset assumptions used by the current model stage.

The goal is to prevent AI assistants or future code changes from guessing the WildTrack directory structure, camera-view mapping, annotation format, or BEV supervision logic.

This file is a contract for:

```text
src/dataset.py
src/geometry.py
src/train_main.py
src/evaluate_main.py
```

Any change to dataset loading, annotation parsing, camera projection, or BEV target generation must remain consistent with this document.

---

## 2. Dataset

The only dataset used in the current model stage is:

```text
WildTrack
```

The current task is:

```text
WildTrack multi-view images
→ BEV pedestrian occupancy heatmap
→ BEV pedestrian point detections
```

The current stage does not use:

```text
synthetic data
real station data
custom unlabeled videos
external tracking datasets
trajectory prediction datasets
```

---

## 3. Expected Root Directory

The dataset root is passed by command line as:

```bash
--data_root wildtrack
```

The expected dataset root contains these core resources:

```text
wildtrack/
├── Image_subsets/
├── annotations_positions/
├── calibrations/
└── rectangles.pom
```

The code must not silently assume a different dataset root structure.

If a path changes, this file and the related code must be updated together.

---

## 4. Image Directory Contract

WildTrack images are stored under:

```text
Image_subsets/
```

Camera folders are expected to follow the WildTrack camera naming convention:

```text
C1
C2
C3
C4
C5
C6
C7
```

The model uses selected camera views from the command line:

```bash
--views 0,1,2
```

The expected mapping is:

```text
view 0 → C1
view 1 → C2
view 2 → C3
view 3 → C4
view 4 → C5
view 5 → C6
view 6 → C7
```

The code must keep this mapping explicit and inspectable.

The code must not reinterpret `view 0` as `C0`.

---

## 5. Annotation Contract

WildTrack ground-plane pedestrian annotations are read from:

```text
annotations_positions/
```

Each annotation file corresponds to one frame.

The annotation provides pedestrian ground-plane positions.

The current model uses these annotations to generate BEV heatmap supervision.

The annotation is not used for:

```text
tracking identity supervision
trajectory prediction supervision
future position prediction
ReID supervision
```

These tasks are outside the current model definition.

---

## 6. Camera Calibration Contract

Camera calibration files are expected under:

```text
calibrations/
```

The current model requires camera projection information for BEV construction.

The model must not ignore calibration or projection information when constructing BEV features.

Projection-related code must be kept in or routed through:

```text
src/geometry.py
```

or another explicitly documented projection module.

Any change to calibration parsing or projection logic must be documented.

---

## 7. BEV Grid Contract

The current model predicts a BEV pedestrian occupancy heatmap.

The BEV grid must be consistent across:

```text
dataset target generation
geometry projection
model output
evaluation
```

The BEV heatmap shape is represented as:

```text
[1, H_bev, W_bev]
```

Where:

```text
H_bev = BEV grid height
W_bev = BEV grid width
```

The exact grid size must be defined by the repository configuration or dataset code.

The model output resolution and ground-truth heatmap resolution must match before loss computation.

---

## 8. Ground Truth Heatmap Contract

Ground-plane pedestrian annotations are converted into BEV heatmap targets.

The heatmap target represents pedestrian occupancy or pedestrian center confidence in BEV space.

The target generation must be deterministic for the same frame and configuration.

The target generation must not depend on model predictions.

The current stage does not generate:

```text
trajectory targets
velocity targets
identity labels
future occupancy targets
crowd risk labels
```

---

## 9. Dataset Sample Contract

Each dataset sample used by training or evaluation should provide:

```text
multi-view images
selected view ids
frame id
BEV ground-truth heatmap
camera projection information or access to it
ground-truth pedestrian BEV points for evaluation, if available
```

The exact Python object structure may vary, but the semantic content must remain stable.

---

## 10. View Selection Contract

The model must support selected camera views.

Example:

```bash
--views 0,1,2
```

The selected views must be used consistently in:

```text
image loading
projection loading
feature projection
multi-view fusion
evaluation metadata
experiment logging
```

An experiment record must include the selected views.

---

## 11. Evaluation Data Contract

Evaluation must compare predicted BEV pedestrian points against WildTrack ground-plane annotations.

Evaluation must not use image-domain bounding boxes as the primary metric for the current stage.

Required evaluation metrics are defined in:

```text
docs/experiment_protocol.md
```

---

## 12. Forbidden Dataset Changes

AI assistants must not:

```text
change the dataset away from WildTrack
reinterpret camera view ids without updating this document
remove calibration usage from BEV projection
use synthetic data in the current stage
introduce trajectory labels in the current stage
introduce ReID labels in the current stage
silently change BEV grid resolution
silently change annotation parsing behavior
```

Any dataset-related change must update this document.

---

## 13. Summary

The current dataset contract is:

```text
WildTrack synchronized multi-view images
+ WildTrack camera calibration
+ WildTrack ground-plane pedestrian annotations
→ BEV pedestrian heatmap supervision
→ BEV pedestrian detection evaluation
```

This contract exists to keep AI-assisted model development aligned with the current BEV pedestrian detection stage.
