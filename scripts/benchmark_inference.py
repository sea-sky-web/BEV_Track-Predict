#!/usr/bin/env python3
"""
GPU inference benchmark: compare fusion modes on latency, FPS, params, FLOPs.

Usage:
    python scripts/benchmark_inference.py --device cuda
    python scripts/benchmark_inference.py --device cpu --warmup 2 --rounds 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import MVDetLikeNet


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def estimate_flops(model: MVDetLikeNet, fusion_mode: str, V: int, H: int, W: int) -> dict:
    """Estimate FLOPs for non-backbone components (fusion + BEV head + offset head)."""
    feat_ch = 512
    Hb, Wb = model.Hb, model.Wb
    hw = Hb * Wb

    fusion_flops = 0
    if fusion_mode == "concat":
        pass
    else:
        # ConcatAttentionFusion / GeoConfidenceFusion
        # joint_compress: Conv2d(V*feat_ch, feat_ch, 1)
        fusion_flops += 2 * V * feat_ch * feat_ch * 1 * hw
        # weight_head / feature_weight_head: Conv2d(feat_ch, V, 1)
        fusion_flops += 2 * feat_ch * V * 1 * hw
        if fusion_mode == "geo_confidence_v1":
            # geo_score_net: Conv2d(3, 1, 1) × V views
            fusion_flops += V * 2 * 3 * 1 * 1 * hw
        # weighted sum: V * feat_ch * hw multiplies + adds
        fusion_flops += 2 * V * feat_ch * hw

    # BEV head
    if fusion_mode == "concat":
        in_ch = V * feat_ch + 2
        head_flops = (
            2 * in_ch * 512 * 9 * hw
            + 2 * 512 * 512 * 9 * hw
            + 2 * 512 * 1 * 9 * hw
        )
    else:
        in_ch = feat_ch + 2
        head_flops = (
            2 * in_ch * 256 * 9 * hw
            + 2 * 256 * 256 * 9 * hw
            + 2 * 256 * 256 * 9 * hw
            + 2 * 256 * 1 * 1 * hw
        )

    # Offset head
    offset_flops = 2 * in_ch * 64 * 9 * hw + 2 * 64 * 2 * 1 * hw

    # Backbone (shared, per view): rough estimate
    # ResNet-18: ~1.8 GFLOPs, MobileNet-V2: ~0.3 GFLOPs per view at 1080x1920
    Hf, Wf = H // 8, W // 8
    scale = (H * W) / (1080 * 1920)
    bb_name = getattr(model, "backbone_name", "resnet18")
    if bb_name == "mobilenet_v2":
        backbone_per_view = 0.3e9 * scale
    else:
        backbone_per_view = 1.8e9 * scale
    backbone_flops = V * backbone_per_view

    # Warp (grid_sample per view)
    warp_flops = V * feat_ch * Hb * Wb * 4  # bilinear = 4 mults per pixel per channel

    return {
        "backbone_gflops": backbone_flops / 1e9,
        "warp_gflops": warp_flops / 1e9,
        "fusion_gflops": fusion_flops / 1e9,
        "head_gflops": head_flops / 1e9,
        "offset_gflops": offset_flops / 1e9,
        "total_gflops": (backbone_flops + warp_flops + fusion_flops + head_flops + offset_flops) / 1e9,
        "non_backbone_gflops": (warp_flops + fusion_flops + head_flops + offset_flops) / 1e9,
    }


def benchmark_model(model, x, warmup=10, rounds=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(rounds):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--views", type=int, default=7)
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    V = args.views
    H, W = args.height, args.width

    proj = torch.randn(V, 3, 3)
    Hf, Wf = H // 8, W // 8
    Hb, Wb = 120, 360

    backbones = ["resnet18", "mobilenet_v2"]
    fusion_modes = ["concat", "confidence_v2", "geo_confidence_v1"]
    results = []

    print(f"Benchmark: {V} views, {H}x{W} input, device={args.device}")
    print(f"Warmup={args.warmup}, Rounds={args.rounds}")
    print("=" * 80)

    for bb in backbones:
      for fm in fusion_modes:
        print(f"\n--- {bb} + {fm} ---")

        model = MVDetLikeNet(
            num_views=V, proj_mats=proj,
            reduced_hw=(Hb, Wb), feat_hw=(Hf, Wf),
            feat_ch=512, pretrained=False, backbone=bb,
            add_coord=True, fusion_mode=fm,
        ).to(device).eval()

        x = torch.randn(1, V, 3, H, W, device=device)

        total_params = count_params(model)
        backbone_params = count_params(model.backbone)
        head_params = count_params(model.bev_head)
        fusion_params = count_params(model.confidence_fusion) if model.confidence_fusion else 0
        non_backbone_params = total_params - backbone_params

        flops = estimate_flops(model, fm, V, H, W)

        times = benchmark_model(model, x, args.warmup, args.rounds, args.device)
        avg_ms = sum(times) / len(times) * 1000
        std_ms = (sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
        fps = 1000.0 / avg_ms

        r = {
            "backbone": bb,
            "fusion_mode": fm,
            "views": V,
            "resolution": f"{H}x{W}",
            "total_params_M": round(total_params / 1e6, 2),
            "backbone_params_M": round(backbone_params / 1e6, 2),
            "non_backbone_params_M": round(non_backbone_params / 1e6, 2),
            "head_params_M": round(head_params / 1e6, 2),
            "fusion_params_M": round(fusion_params / 1e6, 2),
            "total_gflops": round(flops["total_gflops"], 1),
            "non_backbone_gflops": round(flops["non_backbone_gflops"], 1),
            "latency_ms": round(avg_ms, 1),
            "latency_std_ms": round(std_ms, 1),
            "fps": round(fps, 2),
        }
        results.append(r)

        print(f"  Params:  {total_params/1e6:.1f}M total, {non_backbone_params/1e6:.1f}M non-backbone, {head_params/1e6:.1f}M head, {fusion_params/1e6:.2f}M fusion")
        print(f"  GFLOPs:  {flops['total_gflops']:.1f} total, {flops['non_backbone_gflops']:.1f} non-backbone")
        print(f"  Latency: {avg_ms:.1f} ± {std_ms:.1f} ms")
        print(f"  FPS:     {fps:.2f}")

        del model, x
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Scalability analysis
    print("\n" + "=" * 80)
    print("Scalability analysis: params vs num_views")
    print("=" * 80)
    scale_results = []
    for nv in [3, 5, 7, 9, 12]:
        p = torch.randn(nv, 3, 3)
        row = {"views": nv}
        for fm in ["concat", "confidence_v2"]:
            m = MVDetLikeNet(
                num_views=nv, proj_mats=p,
                reduced_hw=(Hb, Wb), feat_hw=(Hf, Wf),
                feat_ch=512, pretrained=False, backbone="resnet18",
                add_coord=True, fusion_mode=fm,
            )
            tp = count_params(m)
            nbp = tp - count_params(m.backbone)
            fl = estimate_flops(m, fm, nv, H, W)
            row[f"{fm}_total_M"] = round(tp / 1e6, 1)
            row[f"{fm}_non_backbone_M"] = round(nbp / 1e6, 1)
            row[f"{fm}_gflops"] = round(fl["total_gflops"], 1)
            del m
        scale_results.append(row)
        print(f"  V={nv:2d}  concat: {row['concat_total_M']:5.1f}M ({row['concat_non_backbone_M']:5.1f}M non-bb, {row['concat_gflops']:7.1f} GF)  "
              f"cv2: {row['confidence_v2_total_M']:5.1f}M ({row['confidence_v2_non_backbone_M']:5.1f}M non-bb, {row['confidence_v2_gflops']:7.1f} GF)")

    all_results = {
        "benchmark": results,
        "scalability": scale_results,
        "config": {"height": H, "width": W, "warmup": args.warmup, "rounds": args.rounds, "device": args.device},
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Backbone':<15s} {'Mode':<20s} {'Params':>8s} {'Non-BB':>8s} {'GFLOPs':>8s} {'ms':>8s} {'FPS':>8s}")
    for r in results:
        print(f"{r['backbone']:<15s} {r['fusion_mode']:<20s} {r['total_params_M']:>7.1f}M {r['non_backbone_params_M']:>7.1f}M {r['non_backbone_gflops']:>7.1f} {r['latency_ms']:>7.1f} {r['fps']:>7.2f}")


if __name__ == "__main__":
    main()
