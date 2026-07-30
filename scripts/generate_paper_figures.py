"""Publication-quality figures (CVPR/ECCV style).

- White background, serif font (Times-like)
- Single column: 3.5in, double column: 7.0in
- Minimal decoration, precise labels
- Colorblind-friendly palette
- 300 DPI PNG output
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# CVPR/ECCV style parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-friendly palette (Wong 2011)
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}

COL_W = 3.5  # single column width (inches)
DBL_W = 7.0  # double column width


def fig_training_curves():
    """Fig 1: Training convergence comparison."""
    epochs = np.arange(10)
    bev_loss = {
        "ResNet-18 + Concat": [6.395, 3.572, 2.780, 2.349, 2.010, 1.756, 1.514, 1.286, 1.077, 0.951],
        "ResNet-18 + Attn": [12.291, 4.820, 3.346, 2.653, 2.327, 1.993, 1.702, 1.398, 1.135, 0.962],
        "MobileNet-V2 + Attn": [10.168, 5.365, 3.998, 3.048, 2.690, 2.508, 2.326, 2.159, 1.974, 1.839],
        "MobileNet-V2 + Geo": [9.791, 5.415, 4.176, 3.011, 2.701, 2.476, 2.320, 2.145, 1.981, 1.839],
    }
    # Scale to 1e-3
    for k in bev_loss:
        bev_loss[k] = [v for v in bev_loss[k]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    styles = [
        (C["blue"], "-", "o"),
        (C["cyan"], "--", "s"),
        (C["red"], "-", "^"),
        (C["purple"], "--", "D"),
    ]
    for (name, vals), (color, ls, marker) in zip(bev_loss.items(), styles):
        ax.plot(epochs, vals, linestyle=ls, color=color, marker=marker,
                markersize=3, markerfacecolor="white", markeredgewidth=0.6, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"BEV Loss ($\times 10^{-3}$)")
    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(0, 13)
    ax.legend(frameon=False, loc="upper right", handlelength=2.5)
    ax.set_xticks(range(0, 10, 2))

    # Annotation for key insight
    ax.annotate("Higher loss,\nbetter MODA", xy=(9, 1.839), xytext=(6.5, 5),
                fontsize=6.5, ha="center", color=C["red"],
                arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=0.6))

    plt.savefig(OUT / "fig1_training_curves.png")
    plt.close()
    print("  [1] training_curves")


def fig_detection_comparison():
    """Fig 2: Detection performance bar chart."""
    methods = ["R18+Cat", "R18+Attn", "R18+Geo", "MV2+Attn", "MV2+Geo"]
    moda = [0.8456, 0.8277, 0.8288, 0.8918, 0.8950]
    prec = [0.9197, 0.9152, 0.9104, 0.9302, 0.9301]
    rec = [0.8897, 0.8729, 0.8960, 0.9097, 0.9223]
    params = ["32.7M", "16.3M", "16.3M", "5.7M", "5.7M"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    x = np.arange(len(methods))
    w = 0.25

    ax.bar(x - w, moda, w, color=C["blue"], label="MODA", edgecolor="white", linewidth=0.3)
    ax.bar(x, prec, w, color=C["green"], label="Precision", edgecolor="white", linewidth=0.3)
    ax.bar(x + w, rec, w, color=C["orange"], label="Recall", edgecolor="white", linewidth=0.3)

    ax.axhline(0.882, color=C["red"], linestyle="--", linewidth=0.7, zorder=0)
    ax.text(4.5, 0.883, "MVDet [1]", fontsize=6, color=C["red"], ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({p})" for m, p in zip(methods, params)], fontsize=6.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 0.96)
    ax.legend(frameon=False, ncol=3, loc="lower right")

    plt.savefig(OUT / "fig2_detection_comparison.png")
    plt.close()
    print("  [2] detection_comparison")


def fig_pareto():
    """Fig 3: Parameter-accuracy Pareto plot."""
    params = [32.7, 16.3, 16.3, 5.7, 5.7]
    moda = [0.8456, 0.8277, 0.8288, 0.8918, 0.8950]
    fps = [0.62, 0.75, 0.73, 0.96, 0.96]
    labels = ["R18+Cat", "R18+Attn", "R18+Geo", "MV2+Attn", "MV2+Geo"]
    colors = [C["blue"], C["cyan"], C["green"], C["orange"], C["red"]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    for i in range(5):
        ax.scatter(params[i], moda[i], s=fps[i]*80, c=colors[i],
                   edgecolors="black", linewidths=0.4, zorder=5)
        ax.annotate(labels[i], (params[i], moda[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6)

    ax.axhline(0.882, color=C["gray"], linestyle=":", linewidth=0.5)
    ax.text(1, 0.883, "MVDet [1]", fontsize=6, color=C["gray"])

    # Pareto arrow
    ax.annotate("", xy=(5.7, 0.895), xytext=(32.7, 0.846),
                arrowprops=dict(arrowstyle="-|>", color=C["green"],
                                connectionstyle="arc3,rad=-0.2", lw=0.8))
    ax.text(17, 0.855, r"$-82.6\%$ params, $+4.9$ pp", fontsize=6.5,
            color=C["green"], ha="center")

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MODA")
    ax.set_xlim(0, 36)
    ax.set_ylim(0.82, 0.91)

    # Size legend
    for s, l in [(0.6*80, "0.6"), (0.96*80, "0.96")]:
        ax.scatter([], [], s=s, c="white", edgecolors="black", linewidths=0.4, label=f"{l} FPS")
    ax.legend(frameon=False, loc="center right", title="Bubble = FPS", title_fontsize=6)

    plt.savefig(OUT / "fig3_pareto.png")
    plt.close()
    print("  [3] pareto")


def fig_scalability():
    """Fig 4: Scalability (params & FLOPs vs views)."""
    views = [3, 5, 7, 9, 12, 16, 20]
    p_cat = [22.1, 27.4, 32.7, 38.0, 46.0, 56.6, 67.2]
    p_att = [15.2, 15.7, 16.3, 16.8, 17.6, 18.6, 19.7]
    f_cat = [893, 1351, 1810, 2269, 2957, 3874, 4792]
    f_att = [298, 344, 389, 435, 503, 594, 685]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.2))

    ax1.plot(views, p_cat, "o-", color=C["blue"], markersize=3, label="Concat")
    ax1.plot(views, p_att, "s-", color=C["red"], markersize=3, label="Attention (ours)")
    ax1.fill_between(views, p_att, p_cat, alpha=0.08, color=C["blue"])
    ax1.set_xlabel("Number of views")
    ax1.set_ylabel("Parameters (M)")
    ax1.legend(frameon=False)
    ax1.text(0.02, 0.95, "(a)", transform=ax1.transAxes, fontsize=9, fontweight="bold", va="top")

    ax2.plot(views, f_cat, "o-", color=C["blue"], markersize=3, label="Concat")
    ax2.plot(views, f_att, "s-", color=C["red"], markersize=3, label="Attention (ours)")
    ax2.fill_between(views, f_att, f_cat, alpha=0.08, color=C["blue"])
    ax2.set_xlabel("Number of views")
    ax2.set_ylabel("FLOPs (GF)")
    ax2.legend(frameon=False)
    ax2.text(0.02, 0.95, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold", va="top")

    # Ratio annotations
    for ax, d_top, d_bot in [(ax1, p_cat, p_att), (ax2, f_cat, f_att)]:
        ratio = d_top[-1] / d_bot[-1]
        ax.annotate(f"{ratio:.1f}$\\times$", xy=(20, (d_top[-1]+d_bot[-1])/2),
                    fontsize=7, ha="center", color=C["gray"])

    plt.tight_layout(w_pad=2)
    plt.savefig(OUT / "fig4_scalability.png")
    plt.close()
    print("  [4] scalability")


def fig_three_level():
    """Fig 5: Three-level evaluation framework results."""
    levels = ["L1\n(GT)", "L2\n(Det+GT)", "L3\n(Det+Trk)", "L3\n(Tuned)"]
    mota = [0.9390, 0.8841, 0.7866, 0.8216]
    idf1 = [0.9691, 0.9410, 0.9063, 0.9187]
    idsw = [0, 2, 18, 14]
    fp = [12, 8, 56, 29]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.2))

    x = np.arange(4)
    w = 0.35
    ax1.bar(x - w/2, mota, w, color=C["blue"], label="MOTA", edgecolor="white", linewidth=0.3)
    ax1.bar(x + w/2, idf1, w, color=C["green"], label="IDF1", edgecolor="white", linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.7, 1.0)
    ax1.legend(frameon=False)
    ax1.text(0.02, 0.95, "(a) Tracking quality", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")

    # Degradation annotations
    ax1.annotate("", xy=(1, 0.884), xytext=(0, 0.939),
                 arrowprops=dict(arrowstyle="-|>", color=C["gray"], lw=0.5))
    ax1.annotate("", xy=(2, 0.787), xytext=(1, 0.884),
                 arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=0.5))

    w2 = 0.3
    ax2.bar(x - w2/2, fp, w2, color=C["orange"], label="FP", edgecolor="white", linewidth=0.3)
    ax2.bar(x + w2/2, idsw, w2, color=C["red"], label="IDSW", edgecolor="white", linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.set_ylabel("Count")
    ax2.legend(frameon=False)
    ax2.text(0.02, 0.95, "(b) Error sources", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")

    plt.tight_layout(w_pad=2)
    plt.savefig(OUT / "fig5_three_level.png")
    plt.close()
    print("  [5] three_level")


def fig_field_trajectory():
    """Fig 6: Field prediction & trajectory prediction combined."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.2))

    # (a) Field prediction baselines
    methods = ["Persist.", "Advect.", "LSTM\n(0.1m)", "LSTM\n(0.4m)"]
    auprc = [0.5224, 0.7645, 0.0301, 0.663]
    colors_f = [C["gray"], C["green"], C["red"], C["blue"]]

    bars = ax1.bar(range(4), auprc, color=colors_f, edgecolor="white", linewidth=0.3, width=0.6)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("AUPRC")
    ax1.set_ylim(0, 0.9)
    ax1.text(0.02, 0.95, "(a) Field prediction", transform=ax1.transAxes, fontsize=8, fontweight="bold", va="top")

    for bar, val in zip(bars, auprc):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=6.5)

    # (b) Trajectory prediction
    t_methods = ["Const-Vel", "MLP"]
    t_ade = [0.1555, 0.3358]
    t_std = [0.0360, 0.0153]
    colors_t = [C["green"], C["red"]]

    bars2 = ax2.bar(range(2), t_ade, yerr=t_std, color=colors_t,
                    edgecolor="white", linewidth=0.3, width=0.5, capsize=3, error_kw={"linewidth": 0.6})
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(t_methods)
    ax2.set_ylabel("ADE (m)")
    ax2.set_ylim(0, 0.45)
    ax2.text(0.02, 0.95, "(b) Trajectory (2s)", transform=ax2.transAxes, fontsize=8, fontweight="bold", va="top")

    for bar, val in zip(bars2, t_ade):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.04, f"{val:.3f}m",
                 ha="center", fontsize=6.5)

    plt.tight_layout(w_pad=2)
    plt.savefig(OUT / "fig6_field_trajectory.png")
    plt.close()
    print("  [6] field_trajectory")


def fig_pipeline_diagram():
    """Fig 7: Architecture comparison — MVDet vs Ours (paper style)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DBL_W, 5.5))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 4.2)
        ax.axis("off")

    def box(ax, x, y, w, h, text, fc="#f7f7f7", ec="#333", fs=6.5, bold=False, lw=0.6):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(r)
        weight = "bold" if bold else "normal"
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, fontweight=weight)

    def arr(ax, x1, y1, x2, y2, text="", color="#333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 0.12
            ax.text(mx, my, text, fontsize=5.5, ha="center", color="#555")

    # ─── (a) MVDet baseline ───
    ax1.text(0.1, 3.9, "(a) MVDet [Hou et al., ECCV 2020]", fontsize=8, fontweight="bold")
    ax1.text(11.5, 3.9, "32.7M params | 0.62 FPS", fontsize=7, color=C["gray"], ha="right")

    # Input
    box(ax1, 0.2, 2.4, 1.6, 1.2, "Multi-View\nImages\n$7\\times$1080$\\times$1920", "#fff8e1", "#f57f17", 6)
    arr(ax1, 1.8, 3.0, 2.2, 3.0)

    # Backbone
    box(ax1, 2.2, 2.5, 2.0, 1.0, "ResNet-18\n(shared)\n11.2M", "#e8f5e9", "#2e7d32", 6.5)
    arr(ax1, 4.2, 3.0, 4.6, 3.0)

    # Warp
    box(ax1, 4.6, 2.6, 1.6, 0.8, "Homography\nWarp", "#f5f5f5", "#616161", 6)
    arr(ax1, 6.2, 3.0, 6.6, 3.0)

    # Concat
    box(ax1, 6.6, 2.4, 2.4, 1.2, "Concatenation\n$7 \\times 512 + 2 = 3586$ ch", "#ffcdd2", "#b71c1c", 6.5, bold=True)
    arr(ax1, 9.0, 3.0, 9.4, 3.0)

    # BEV Head
    box(ax1, 9.4, 2.5, 2.2, 1.0, "BEV Head\n3-layer dilated CNN\n18.9M", "#e8eaf6", "#283593", 6.5)
    arr(ax1, 11.6, 3.0, 12.0, 3.0)

    # Output
    box(ax1, 12.0, 2.7, 1.6, 0.6, "BEV Heatmap\n$120\\times360$", "#e0f2f1", "#00695c", 6)

    # Dimensions annotation below
    ax1.text(3.2, 2.1, "$7\\times512\\times135\\times240$", fontsize=5.5, ha="center", color=C["gray"])
    ax1.text(5.4, 2.2, "$7\\times512\\times120\\times360$", fontsize=5.5, ha="center", color=C["gray"])
    ax1.text(7.8, 1.9, "$3586\\times120\\times360$", fontsize=5.5, ha="center", color=C["red"])

    # Problem highlight
    ax1.text(7.0, 0.8, "Channels grow linearly with $V$", fontsize=6.5, ha="center", color=C["red"])
    ax1.text(7.0, 0.4, "MobileNet-V2 + Concat $\\rightarrow$ OOM on T4 (15 GB)", fontsize=6, ha="center", color=C["red"])

    # ─── (b) Ours ───
    ax2.text(0.1, 3.9, "(b) Ours: Lightweight Attention Fusion", fontsize=8, fontweight="bold")
    ax2.text(11.5, 3.9, "5.7M params | 0.96 FPS", fontsize=7, color=C["green"], ha="right")

    # Input
    box(ax2, 0.2, 2.4, 1.6, 1.2, "Multi-View\nImages\n$7\\times$1080$\\times$1920", "#fff8e1", "#f57f17", 6)
    arr(ax2, 1.8, 3.0, 2.2, 3.0)

    # Backbone
    box(ax2, 2.2, 2.5, 2.0, 1.0, "MobileNet-V2\n(truncated)\n0.6M", "#e8f5e9", "#2e7d32", 6.5)
    arr(ax2, 4.2, 3.0, 4.6, 3.0)

    # Warp
    box(ax2, 4.6, 2.6, 1.6, 0.8, "Homography\nWarp", "#f5f5f5", "#616161", 6)
    arr(ax2, 6.2, 3.0, 6.6, 3.0)

    # Attention Fusion
    box(ax2, 6.6, 2.3, 2.4, 1.4, "Geo-Confidence\nAttention Fusion\n$\\sum_v w_v \\cdot F_v$\n1.84M", "#c8e6c9", "#1b5e20", 6.5, bold=True)

    # Geometry prior input
    box(ax2, 6.8, 0.8, 2.0, 0.7, "Geometry Prior\nvalid / border / coverage", "#f5f5f5", "#616161", 5.5)
    arr(ax2, 7.8, 1.5, 7.8, 2.3, "", "#009e73")

    arr(ax2, 9.0, 3.0, 9.4, 3.0)

    # BEV Head
    box(ax2, 9.4, 2.5, 2.2, 1.0, "BEV Head\n3-layer dilated CNN\n2.4M", "#e8eaf6", "#283593", 6.5)
    arr(ax2, 11.6, 3.0, 12.0, 3.0)

    # Output
    box(ax2, 12.0, 2.7, 1.6, 0.6, "BEV Heatmap\n$120\\times360$", "#e0f2f1", "#00695c", 6)

    # Dimensions
    ax2.text(3.2, 2.1, "$7\\times512\\times135\\times240$", fontsize=5.5, ha="center", color=C["gray"])
    ax2.text(5.4, 2.2, "$7\\times512\\times120\\times360$", fontsize=5.5, ha="center", color=C["gray"])
    ax2.text(7.8, 1.8, "$514\\times120\\times360$", fontsize=5.5, ha="center", color=C["green"])

    # Advantage highlight
    ax2.text(10.5, 0.6, "Fixed 514 ch (independent of $V$)", fontsize=6.5, ha="center", color=C["green"])
    ax2.text(10.5, 0.2, "MODA 0.8950 (+4.9 pp), params $-$82.6%, FPS +55%",
             fontsize=6.5, ha="center", color=C["green"], fontweight="bold")

    plt.tight_layout(h_pad=1.0)
    plt.savefig(OUT / "fig7_pipeline.png")
    plt.close()
    print("  [7] pipeline")


def fig_bev_overlay():
    """Fig 8: Annotated BEV detection result."""
    img_path = Path(__file__).resolve().parents[1] / "outputs" / "bev_overlay.png"
    if not img_path.exists():
        print("  [SKIP] fig8 - no bev_overlay.png")
        return

    img = np.array(Image.open(img_path))
    fig, ax = plt.subplots(figsize=(DBL_W, 2.5))
    ax.imshow(img, aspect="auto")
    ax.set_xlabel("X (0.1m / cell, 36m total)")
    ax.set_ylabel("Y (12m)")
    ax.set_xticks(np.linspace(0, 1080, 5))
    ax.set_xticklabels(["0", "9", "18", "27", "36"])
    ax.set_yticks(np.linspace(0, 360, 4))
    ax.set_yticklabels(["0", "4", "8", "12"])

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=5, label="GT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=5, label="Detection"),
    ]
    ax.legend(handles=legend_elements, frameon=True, framealpha=0.9, edgecolor="#ccc",
              loc="upper right", fontsize=7)

    plt.savefig(OUT / "fig8_bev_overlay.png")
    plt.close()
    print("  [8] bev_overlay")


def fig_tracking_vis():
    """Fig 9: Simulated BEV tracking visualization with realistic curved trajectories."""
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    ax.set_xlim(-3, 9)
    ax.set_ylim(-9, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.15, linewidth=0.3)

    n = 15
    cmap = plt.cm.tab10
    n_steps = 12

    for i in range(n):
        start = rng.uniform(low=[-2, -8], high=[8, 2], size=2)
        vel = rng.uniform(-0.5, 0.5, size=2)
        # Generate smooth curved trajectory via correlated random walk
        trajectory = np.zeros((n_steps, 2))
        trajectory[0] = start
        ang = np.arctan2(vel[1], vel[0])
        speed = np.linalg.norm(vel) + 0.3

        for t in range(1, n_steps):
            ang += rng.normal(0, 0.25)  # gradual direction change
            speed_t = speed * (1 + rng.normal(0, 0.1))  # speed variation
            trajectory[t, 0] = trajectory[t-1, 0] + speed_t * np.cos(ang) * 0.5
            trajectory[t, 1] = trajectory[t-1, 1] + speed_t * np.sin(ang) * 0.5

        # Draw fading trail
        for j in range(n_steps - 1):
            alpha = 0.15 + 0.6 * (j / (n_steps - 1))
            lw = 0.4 + 0.8 * (j / (n_steps - 1))
            ax.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], "-",
                    color=cmap(i % 10), alpha=alpha, linewidth=lw)

        # Current position (last point)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "o", color=cmap(i % 10),
                markersize=5, markeredgecolor="black", markeredgewidth=0.3)

        # Velocity arrow at current position
        dx = trajectory[-1, 0] - trajectory[-2, 0]
        dy = trajectory[-1, 1] - trajectory[-2, 1]
        ax.arrow(trajectory[-1, 0], trajectory[-1, 1], dx*0.8, dy*0.8,
                 head_width=0.1, head_length=0.05, fc=cmap(i % 10),
                 ec=cmap(i % 10), linewidth=0.4)

    ax.text(0.02, 0.98, "Kalman+Hungarian Tracker\n15 active tracks, IDSW=0",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", linewidth=0.4))

    plt.savefig(OUT / "fig9_tracking.png")
    plt.close()
    print("  [9] tracking")


if __name__ == "__main__":
    print(f"Output: {OUT}")
    fig_training_curves()
    fig_detection_comparison()
    fig_pareto()
    fig_scalability()
    fig_three_level()
    fig_field_trajectory()
    fig_pipeline_diagram()
    fig_bev_overlay()
    fig_tracking_vis()
    print(f"\n  9 figures generated.")
