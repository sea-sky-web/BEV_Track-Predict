#!/usr/bin/env python
"""
Quick module integrity checks for the refactored `src/` package.

This script intentionally keeps output ASCII-only so it works in
Windows terminals configured with GBK code pages.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _ensure_src_on_path() -> None:
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _check_import(module_name: str, required_attrs: list[str]) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        missing = [name for name in required_attrs if not hasattr(module, name)]
        if missing:
            return False, f"missing attrs: {', '.join(missing)}"
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def verify_all_modules() -> int:
    _ensure_src_on_path()

    print("=" * 60)
    print("Module Integrity Check (src/)")
    print("=" * 60)

    checks: list[tuple[str, list[str]]] = [
        ("config", ["NB_WIDTH", "STEP_M"]),
        ("utils", ["build_gaussian_kernel_2d", "save_heat_png"]),
        ("loss", ["GaussianMSE", "create_loss_criterion"]),
        ("augmentation", ["ViewCoherentAugment", "parse_color_jitter"]),
        ("metrics", ["compute_moda_modp", "aggregate_metrics"]),
        ("calibration", ["parse_rectangles_pom", "decide_unit_scale", "CalibrationLoader"]),
        ("geometry", ["make_worldgrid2worldcoord_mat", "compute_valid_ratio_from_homography"]),
        ("models", ["ResNet18Stride8Trunk", "ResNet50Stride8Trunk", "MVDetLikeNet", "create_model"]),
        ("dataset", ["WildtrackMVDetDataset", "create_wildtrack_dataset"]),
        ("trainer", ["MVDetTrainer", "create_optimizer", "create_scheduler"]),
        ("train_main", ["parse_args", "main"]),
        ("evaluate_main", ["parse_args", "main"]),
    ]

    passed = 0
    for index, (module_name, attrs) in enumerate(checks, start=1):
        ok, detail = _check_import(module_name, attrs)
        status = "PASS" if ok else "FAIL"
        print(f"[{index:02d}/{len(checks):02d}] {module_name:<14} {status} - {detail}")
        if ok:
            passed += 1

    print("=" * 60)
    print(f"Result: {passed}/{len(checks)} modules passed")
    print("=" * 60)

    if passed == len(checks):
        print("All checks passed.")
        return 0

    print("Some checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(verify_all_modules())
