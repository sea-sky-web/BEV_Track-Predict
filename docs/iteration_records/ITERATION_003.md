# ITERATION_003 Diagnostic Record

## Previous run

`ai_runs/latest_run.txt` points to `20260511_041025`.

The run completed successfully with `success=true` and `return_code=0`, but the detection metrics are not useful for model optimization:

- Precision: `0.09330651469437926`
- Recall: `0.11084561263772734`
- F1: `0.10132265501759496`
- False positives: `8114`
- Missed detections: `6698`
- Localization error: `0.19468057917977521 m`
- Best detection threshold: `0.1`
- Loss: `0.014492000409712394`
- Fusion mode: `confidence`
- Alpha: `2.0`

## Input evidence

Read evidence:

- `ai_runs/latest_run.txt`
- `ai_runs/20260511_041025/metrics.json`
- `src/geometry.py`
- `src/dataset.py`
- `src/evaluate_main.py`
- MVDet reference implementation from `hou-yz/MVDet`
- Colab BEV sanity diagnostic over the first 20 WildTrack frames

The Colab diagnostic checked annotation counts, BEV GT heatmap counts, GT self-evaluation, image auxiliary heatmap alignment, and image-to-BEV homography consistency.

## Model understanding

The model encodes each camera view into an image feature map, then calls `warp_perspective_torch` to sample each source feature map into a reduced BEV feature map. The warp matrix is documented and consumed as `M_src2dst`, where source coordinates are image feature coordinates `(u, v)` and destination coordinates are BEV grid coordinates `(grid_x, grid_y)`.

The dataset and evaluation logic store BEV heatmaps as `(H=grid_y, W=grid_x)`, with WildTrack `positionID` decoded as `grid_x = positionID % NB_WIDTH` and `grid_y = positionID // NB_WIDTH`.

MVDet's original WildTrack code applies a permutation because its map tensor uses `worldgrid_shape=[480,1440]`, effectively placing `grid_x` on the row dimension. This repository uses the conventional tensor layout `(NB_HEIGHT=1440, NB_WIDTH=480)`, so copying the MVDet permutation swaps x/y for this codebase.

## Observed problem

The sanity diagnostic showed that the current projection matrix maps image feature points to swapped BEV coordinates:

- `proj_image_to_bev_expected_xy_err` median: `205.4145` BEV cells
- `proj_image_to_bev_swapped_yx_err` median: approximately `7.65e-14` BEV cells
- `inv_bev_xy_to_image_err_px` median: `206.0719` pixels
- `inv_bev_yx_to_image_err_px` median: approximately `3.69e-13` pixels

Example:

```text
expected_xy: [86.125, 237.375]
projected_by_current_matrix: [237.375, 86.125]
```

At this stage the low precision/recall cannot be interpreted as a model-capacity issue, because multi-view features are being warped into the wrong BEV coordinate convention.

## Candidate causes

1. MVDet permutation was copied without adapting to this repository's BEV tensor layout.
   - Supporting evidence: MVDet uses `worldgrid_shape=[480,1440]`; this repository uses `(NB_HEIGHT=1440, NB_WIDTH=480)`.
   - Minimal validation action: remove the permutation and test that image `(x,y)` maps to BEV `(x,y)`, not `(y,x)`.

2. Dataset GT heatmap may be transposed.
   - Evidence against: GT self-evaluation over 20 frames is perfect, and raw object counts match raw/pool/extracted GT counts.
   - Minimal validation action: keep dataset unchanged.

3. Evaluation matching may be transposed.
   - Evidence against: pooled GT evaluated against itself gives `Precision=1.0`, `Recall=1.0`, `F1=1.0`.
   - Minimal validation action: keep evaluation unchanged.

## Selected hypothesis

The projection matrix should return image feature coordinates to BEV `(grid_x, grid_y)` directly for this repository. The MVDet permutation should not be applied because this repository's BEV tensor already stores row as `grid_y` and column as `grid_x`.

## Rejected options

- Do not change model architecture.
- Do not change detection thresholds or matching distance.
- Do not change dataset annotations or GT heatmap generation.
- Do not add distance suppression or another postprocessing step.
- Do not run another ALPHA/fusion experiment before geometry sanity passes.

## Change boundary

This iteration is limited to:

- `src/geometry.py`: remove the MVDet-style x/y permutation from `build_mvdet_proj_mat` and document why this repository's tensor convention differs from MVDet.
- `tests/test_geometry.py`: add a synthetic projection test that fails if BEV x/y are swapped.

No training, evaluation metric definition, dataset file, or ai_runs history is modified.

## Input output acceptance criteria

Input:

- A simple synthetic camera where image feature coordinates equal world grid coordinates after the known half-cell worldgrid offset.

Output:

- `build_mvdet_proj_mat` maps image `(13.5, 27.5)` to BEV `(13.0, 27.0)`.
- The inverse matrix maps BEV `(13.0, 27.0)` back to image `(13.5, 27.5)`.
- The projection does not map to swapped BEV `(27.0, 13.0)`.

Acceptance criteria:

- `PYTHONPYCACHEPREFIX=/private/tmp/bev_pycache python3 -m compileall src scripts tests` passes.
- The geometry test functions in `tests/test_geometry.py` pass.
- Next Colab validation should rerun the 20-frame BEV sanity diagnostic and show near-zero `expected_xy` error, not near-zero `swapped_yx` error.

## Expected metric impact

No metric improvement is claimed until retraining. The expected immediate impact is that multi-view image features are sampled into the correct BEV coordinates, making future training metrics interpretable.

## Validation command

```bash
PYTHONPYCACHEPREFIX=/private/tmp/bev_pycache python3 -m compileall src scripts tests
PYTHONPYCACHEPREFIX=/private/tmp/bev_pycache python3 - <<'PY'
import tests.test_geometry as tg
tg.test_build_mvdet_proj_mat_preserves_bev_xy_order()
tg.test_build_mvdet_proj_mat_inverse_maps_bev_xy_to_image_xy()
print('geometry tests passed')
PY
```

Final Colab validation:

```bash
python scripts/run_colab_exp.py
python scripts/commit_ai_runs.py
```

Only run training after the BEV sanity diagnostic confirms the projection convention is fixed.

## Rollback plan

Revert this PR. The rollback restores the previous MVDet-style permutation in `src/geometry.py` and removes the synthetic geometry tests. No ai_runs history or dataset files are touched.

## Next forbidden directions

- Do not implement tracking.
- Do not implement ReID.
- Do not implement trajectory prediction.
- Do not introduce occupancy-flow or crowd forecasting.
- Do not introduce BEVFormer, PETR, LSS, DETR3D, or other large BEV frameworks.
- Do not optimize ALPHA/fusion until projection sanity is verified on WildTrack data.
