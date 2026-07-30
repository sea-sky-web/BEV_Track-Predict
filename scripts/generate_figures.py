"""Generate all result visualization figures for BEV_Track-Predict.

Produces publication-quality plots from archived numerical data.
Output: docs/figures/*.png at 300 DPI.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "r18_concat": "#1f77b4",
    "r18_cv2": "#ff7f0e",
    "r18_geo": "#2ca02c",
    "mv2_cv2": "#d62728",
    "mv2_geo": "#9467bd",
}


# ═══════════════════════════════════════════════════════
# DATA (from module1_experiment_data_archive.md)
# ═══════════════════════════════════════════════════════

EPOCHS = list(range(10))

BEV_LOSS = {
    "ResNet-18 + concat": [0.006395, 0.003572, 0.002780, 0.002349, 0.002010, 0.001756, 0.001514, 0.001286, 0.001077, 0.000951],
    "ResNet-18 + cv2": [0.012291, 0.004820, 0.003346, 0.002653, 0.002327, 0.001993, 0.001702, 0.001398, 0.001135, 0.000962],
    "ResNet-18 + geo_cv1": [0.011722, 0.004822, 0.003287, 0.002684, 0.002300, 0.001988, 0.001692, 0.001388, 0.001121, 0.000947],
    "MobileNet-V2 + cv2": [0.010168, 0.005365, 0.003998, 0.003048, 0.002690, 0.002508, 0.002326, 0.002159, 0.001974, 0.001839],
    "MobileNet-V2 + geo_cv1": [0.009791, 0.005415, 0.004176, 0.003011, 0.002701, 0.002476, 0.002320, 0.002145, 0.001981, 0.001839],
}

CONFIGS = ["ResNet-18\nconcat", "ResNet-18\ncv2", "ResNet-18\ngeo_cv1", "MobileNet-V2\ncv2", "MobileNet-V2\ngeo_cv1"]
MODA = [0.8456, 0.8277, 0.8288, 0.8918, 0.8950]
PRECISION = [0.9197, 0.9152, 0.9104, 0.9302, 0.9301]
RECALL = [0.8897, 0.8729, 0.8960, 0.9097, 0.9223]
F1 = [0.9044, 0.8935, 0.9031, 0.9198, 0.9262]
PARAMS_M = [32.7, 16.3, 16.3, 5.7, 5.7]
FPS = [0.62, 0.75, 0.73, 0.96, 0.96]

PARAM_BREAKDOWN = {
    "Backbone": [11.18, 11.18, 11.18, 0.60, 0.60],
    "Fusion": [0, 1.84, 1.84, 1.84, 1.84],
    "BEV Head": [18.89, 2.37, 2.37, 2.37, 2.37],
    "Other": [2.66, 0.89, 0.89, 0.89, 0.89],
}

VIEWS = [3, 5, 7, 9, 12, 16, 20]
PARAMS_CONCAT = [22.1, 27.4, 32.7, 38.0, 46.0, 56.6, 67.2]
PARAMS_ATTN = [15.2, 15.7, 16.3, 16.8, 17.6, 18.6, 19.7]
FLOPS_CONCAT = [893, 1351, 1810, 2269, 2957, 3874, 4792]
FLOPS_ATTN = [298, 344, 389, 435, 503, 594, 685]

# M2 data
L_LABELS = ["L1\n(GT+GT)", "L2\n(Det+GT)", "L3\n(Det+Trk)", "L3\n(Optimized)"]
L_MOTA = [0.9390, 0.8841, 0.7866, 0.8216]
L_IDF1 = [0.9691, 0.9410, 0.9063, 0.9187]
L_IDSW = [0, 2, 18, 14]
L_FP = [12, 8, 56, 29]
L_FN = [28, 66, 66, 74]

FIELD_METHODS = ["Persistence", "Advection", "ConvLSTM\n(bev_down=4)", "ConvLSTM\n(bev_down=16)"]
FIELD_AUPRC = [0.5224, 0.7645, 0.0301, 0.663]

TRAJ_METHODS = ["Constant\nVelocity", "MLP\n(5 seeds)"]
TRAJ_ADE = [0.1555, 0.3358]
TRAJ_STD = [0.0360, 0.0153]

MILESTONES = {
    "06-29": (0.0, "3 bugs found"),
    "06-30": (0.529, "First MODA>0"),
    "07-02": (0.441, "No data leak"),
    "07-06": (0.793, "NMS fix"),
    "07-07": (0.857, "Grid scan"),
    "07-13": (0.8456, "Unified exp"),
    "07-14": (0.8950, "Best"),
}


# ═══════════════════════════════════════════════════════
# FIGURE 1: Training Loss Curves
# ═══════════════════════════════════════════════════════
def fig1_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for name, color in [("ResNet-18 + concat", COLORS["r18_concat"]),
                         ("ResNet-18 + cv2", COLORS["r18_cv2"]),
                         ("ResNet-18 + geo_cv1", COLORS["r18_geo"])]:
        ax1.plot(EPOCHS, BEV_LOSS[name], "o-", color=color, markersize=4, label=name)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BEV Loss")
    ax1.set_title("(a) ResNet-18 Backbone")
    ax1.legend()
    ax1.set_ylim(0, 0.013)
    ax1.grid(True, alpha=0.3)

    for name, color in [("MobileNet-V2 + cv2", COLORS["mv2_cv2"]),
                         ("MobileNet-V2 + geo_cv1", COLORS["mv2_geo"])]:
        ax2.plot(EPOCHS, BEV_LOSS[name], "o-", color=color, markersize=4, label=name)
    ax2.plot(EPOCHS, BEV_LOSS["ResNet-18 + concat"], "--", color=COLORS["r18_concat"],
             alpha=0.5, label="ResNet-18+concat (ref)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BEV Loss")
    ax2.set_title("(b) MobileNet-V2 Backbone")
    ax2.legend()
    ax2.set_ylim(0, 0.013)
    ax2.grid(True, alpha=0.3)
    ax2.annotate("Higher loss,\nbetter MODA!", xy=(9, 0.001839), fontsize=8,
                 xytext=(7, 0.004), arrowprops=dict(arrowstyle="->", color="red"),
                 color="red", ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "fig1_training_curves.png")
    plt.close()
    print("  ✓ fig1_training_curves.png")


# ═══════════════════════════════════════════════════════
# FIGURE 2: Detection Performance Comparison
# ═══════════════════════════════════════════════════════
def fig2_detection_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONFIGS))
    w = 0.2

    bars_moda = ax.bar(x - 1.5*w, MODA, w, label="MODA", color="#2196F3")
    bars_p = ax.bar(x - 0.5*w, PRECISION, w, label="Precision", color="#4CAF50")
    bars_r = ax.bar(x + 0.5*w, RECALL, w, label="Recall", color="#FF9800")
    bars_f1 = ax.bar(x + 1.5*w, F1, w, label="F1", color="#9C27B0")

    ax.axhline(y=0.882, color="red", linestyle="--", linewidth=1.5, label="MVDet paper (0.882)")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS)
    ax.set_ylabel("Score")
    ax.set_ylim(0.78, 0.96)
    ax.set_title("Module 1: Detection Performance Comparison")
    ax.legend(loc="lower right", ncol=3)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars_moda:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_detection_comparison.png")
    plt.close()
    print("  ✓ fig2_detection_comparison.png")


# ═══════════════════════════════════════════════════════
# FIGURE 3: Parameter Efficiency
# ═══════════════════════════════════════════════════════
def fig3_parameter_efficiency():
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLORS["r18_concat"], COLORS["r18_cv2"], COLORS["r18_geo"],
              COLORS["mv2_cv2"], COLORS["mv2_geo"]]
    labels = ["R18+concat", "R18+cv2", "R18+geo", "MV2+cv2", "MV2+geo"]

    for i in range(5):
        ax.scatter(PARAMS_M[i], MODA[i], s=FPS[i]*300, c=colors[i],
                   alpha=0.7, edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(labels[i], (PARAMS_M[i], MODA[i]),
                    textcoords="offset points", xytext=(5, 8), fontsize=8)

    ax.axhline(y=0.882, color="red", linestyle="--", alpha=0.7, label="MVDet paper")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("MODA")
    ax.set_title("Parameter Efficiency (bubble size ∝ FPS)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 36)
    ax.set_ylim(0.82, 0.91)

    ax.annotate("", xy=(5.7, 0.895), xytext=(32.7, 0.846),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(18, 0.86, "−82.6% params\n+4.9pp MODA", fontsize=9, color="green", ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "fig3_parameter_efficiency.png")
    plt.close()
    print("  ✓ fig3_parameter_efficiency.png")


# ═══════════════════════════════════════════════════════
# FIGURE 4: Speed vs Accuracy
# ═══════════════════════════════════════════════════════
def fig4_speed_accuracy():
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [COLORS["r18_concat"], COLORS["r18_cv2"], COLORS["r18_geo"],
              COLORS["mv2_cv2"], COLORS["mv2_geo"]]
    labels = ["R18+concat", "R18+cv2", "R18+geo", "MV2+cv2", "MV2+geo"]

    for i in range(5):
        ax.scatter(FPS[i], MODA[i], s=120, c=colors[i], edgecolors="black",
                   linewidths=0.5, zorder=5)
        ax.annotate(labels[i], (FPS[i], MODA[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(y=0.882, color="red", linestyle="--", alpha=0.7, label="MVDet paper")
    ax.set_xlabel("FPS (T4 GPU)")
    ax.set_ylabel("MODA")
    ax.set_title("Speed vs Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_speed_accuracy.png")
    plt.close()
    print("  ✓ fig4_speed_accuracy.png")


# ═══════════════════════════════════════════════════════
# FIGURE 5: Scalability
# ═══════════════════════════════════════════════════════
def fig5_scalability():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(VIEWS, PARAMS_CONCAT, "o-", color=COLORS["r18_concat"], label="Concat")
    ax1.plot(VIEWS, PARAMS_ATTN, "s-", color=COLORS["mv2_geo"], label="Attention")
    ax1.fill_between(VIEWS, PARAMS_ATTN, PARAMS_CONCAT, alpha=0.1, color="gray")
    ax1.set_xlabel("Number of Views")
    ax1.set_ylabel("Parameters (M)")
    ax1.set_title("(a) Parameters vs Views")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ratio_p = [c/a for c, a in zip(PARAMS_CONCAT, PARAMS_ATTN)]
    for i, v in enumerate(VIEWS):
        ax1.annotate(f"{ratio_p[i]:.1f}×", (v, (PARAMS_CONCAT[i]+PARAMS_ATTN[i])/2),
                     fontsize=7, ha="center", color="gray")

    ax2.plot(VIEWS, FLOPS_CONCAT, "o-", color=COLORS["r18_concat"], label="Concat")
    ax2.plot(VIEWS, FLOPS_ATTN, "s-", color=COLORS["mv2_geo"], label="Attention")
    ax2.fill_between(VIEWS, FLOPS_ATTN, FLOPS_CONCAT, alpha=0.1, color="gray")
    ax2.set_xlabel("Number of Views")
    ax2.set_ylabel("FLOPs (GF)")
    ax2.set_title("(b) FLOPs vs Views")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_scalability.png")
    plt.close()
    print("  ✓ fig5_scalability.png")


# ═══════════════════════════════════════════════════════
# FIGURE 6: Parameter Breakdown
# ═══════════════════════════════════════════════════════
def fig6_param_breakdown():
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["R18+concat\n(32.7M)", "R18+cv2\n(16.3M)", "R18+geo\n(16.3M)",
              "MV2+cv2\n(5.7M)", "MV2+geo\n(5.7M)"]
    x = np.arange(5)
    colors = ["#1976D2", "#F57C00", "#388E3C", "#7B1FA2"]
    bottom = np.zeros(5)

    for i, (comp, vals) in enumerate(PARAM_BREAKDOWN.items()):
        ax.bar(x, vals, bottom=bottom, label=comp, color=colors[i], edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Parameters (M)")
    ax.set_title("Model Component Parameter Breakdown")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_param_breakdown.png")
    plt.close()
    print("  ✓ fig6_param_breakdown.png")


# ═══════════════════════════════════════════════════════
# FIGURE 7: Three-Level Evaluation
# ═══════════════════════════════════════════════════════
def fig7_three_level():
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(4)
    w = 0.3

    bars1 = ax1.bar(x - w/2, L_MOTA, w, label="MOTA", color="#2196F3", edgecolor="white")
    bars2 = ax1.bar(x + w/2, L_IDF1, w, label="IDF1", color="#4CAF50", edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(L_LABELS)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.7, 1.0)
    ax1.set_title("Module 2: Three-Level Evaluation Framework")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, L_IDSW, "D--", color="red", markersize=8, label="ID Switches")
    ax2.plot(x, L_FP, "^--", color="orange", markersize=8, label="False Positives")
    ax2.set_ylabel("Count", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 65)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.annotate("Detection\nerror", xy=(1, 0.88), xytext=(0.5, 0.75),
                 arrowprops=dict(arrowstyle="->"), fontsize=8, ha="center")
    ax1.annotate("Tracker\nerror", xy=(2, 0.79), xytext=(2.5, 0.73),
                 arrowprops=dict(arrowstyle="->"), fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(OUT / "fig7_three_level_eval.png")
    plt.close()
    print("  ✓ fig7_three_level_eval.png")


# ═══════════════════════════════════════════════════════
# FIGURE 8: Field Prediction
# ═══════════════════════════════════════════════════════
def fig8_field_prediction():
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#90A4AE", "#4CAF50", "#F44336", "#2196F3"]
    bars = ax.bar(FIELD_METHODS, FIELD_AUPRC, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, FIELD_AUPRC):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random baseline")
    ax.set_ylabel("Validation AUPRC")
    ax.set_title("Module 2: Field Prediction Baselines (Occupancy)")
    ax.set_ylim(0, 0.9)
    ax.grid(True, axis="y", alpha=0.3)

    ax.annotate("FAILED\n(sparse signal)", xy=(2, 0.03), xytext=(2, 0.15),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=8, ha="center", color="red")
    ax.annotate("Low-res\nworks!", xy=(3, 0.663), xytext=(3, 0.78),
                fontsize=8, ha="center", color="blue")

    plt.tight_layout()
    plt.savefig(OUT / "fig8_field_prediction.png")
    plt.close()
    print("  ✓ fig8_field_prediction.png")


# ═══════════════════════════════════════════════════════
# FIGURE 9: Trajectory Prediction
# ═══════════════════════════════════════════════════════
def fig9_trajectory():
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(TRAJ_METHODS, TRAJ_ADE, yerr=TRAJ_STD, color=colors,
                  edgecolor="black", linewidth=0.5, capsize=5)

    for bar, val in zip(bars, TRAJ_ADE):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.04, f"{val:.4f}m",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("ADE (meters)")
    ax.set_title("Trajectory Prediction (2s horizon)")
    ax.set_ylim(0, 0.45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.annotate("MLP 2.2x worse\nNegative experiment",
                xy=(1, 0.34), xytext=(1, 0.42), fontsize=9, ha="center", color="red")

    plt.tight_layout()
    plt.savefig(OUT / "fig9_trajectory_prediction.png")
    plt.close()
    print("  ✓ fig9_trajectory_prediction.png")


# ═══════════════════════════════════════════════════════
# FIGURE 10: Error Decomposition Waterfall
# ═══════════════════════════════════════════════════════
def fig10_error_waterfall():
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["L1 MOTA\n(baseline)", "Det: +FN", "Trk: +FP", "Trk: +IDSW", "L3 MOTA\n(result)"]
    values = [0.939, -(66-28)/656, -(56-8)/656, -(18-2)/656, 0]
    values[-1] = 0.939 + sum(values[1:-1])

    running = [0.939]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    running.append(values[-1])

    colors = ["#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#2196F3"]
    bottoms = [0, running[0]+values[1], running[1]+values[2], running[2]+values[3], 0]

    for i in range(5):
        if i == 0 or i == 4:
            ax.bar(i, running[i], color=colors[i], edgecolor="black", linewidth=0.5)
        else:
            ax.bar(i, abs(values[i]), bottom=running[i], color=colors[i],
                   edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(categories)
    ax.set_ylabel("MOTA")
    ax.set_title("Module 2: Error Decomposition (L1 → L3)")
    ax.set_ylim(0.7, 0.96)
    ax.grid(True, axis="y", alpha=0.3)

    ax.text(1, running[1]-0.01, f"−{(66-28)/656:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.text(2, running[2]-0.01, f"−{(56-8)/656:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.text(3, running[3]-0.01, f"−{(18-2)/656:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "fig10_error_waterfall.png")
    plt.close()
    print("  ✓ fig10_error_waterfall.png")


# ═══════════════════════════════════════════════════════
# FIGURE 11: MODA Progress Timeline
# ═══════════════════════════════════════════════════════
def fig11_timeline():
    fig, ax = plt.subplots(figsize=(10, 5))

    dates = list(MILESTONES.keys())
    modas = [v[0] for v in MILESTONES.values()]
    notes = [v[1] for v in MILESTONES.values()]

    ax.plot(range(len(dates)), modas, "o-", color="#2196F3", markersize=8, linewidth=2)

    for i, (d, m, n) in enumerate(zip(dates, modas, notes)):
        offset = 0.03 if i % 2 == 0 else -0.06
        ax.annotate(f"{n}\n({m:.3f})", (i, m), textcoords="offset points",
                    xytext=(0, 15 if offset > 0 else -25), fontsize=7.5,
                    ha="center", arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    ax.axhline(y=0.882, color="red", linestyle="--", alpha=0.7, label="MVDet paper (0.882)")
    ax.axhspan(-0.05, 0.0, xmin=0, xmax=0.02, alpha=0.1, color="red")
    ax.text(-0.3, 0.0, "56 failed runs\n(May-Jun)", fontsize=8, color="red", ha="left")

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30)
    ax.set_xlabel("Date (2026)")
    ax.set_ylabel("MODA")
    ax.set_title("Module 1: MODA Progress Timeline")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.95)

    plt.tight_layout()
    plt.savefig(OUT / "fig11_moda_timeline.png")
    plt.close()
    print("  ✓ fig11_moda_timeline.png")


# ═══════════════════════════════════════════════════════
# FIGURE 12: Tracker Parameter Effect
# ═══════════════════════════════════════════════════════
def fig12_tracker_params():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    configs = ["Baseline\n(2,2,1.0)", "#1\n(2,1,0.75)", "#2\n(2,1,1.0)", "#3\n(3,1,1.0)", "#4\n(3,1,0.75)"]
    mota_vals = [0.7866, 0.8216, 0.8171, 0.8095, 0.8064]
    fp_vals = [56, 29, 33, 16, 16]
    fn_vals = [66, 74, 73, 102, 102]
    idsw_vals = [18, 14, 14, 7, 9]

    x = np.arange(5)
    ax1.bar(x, mota_vals, color=["#F44336"] + ["#4CAF50"]*4, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.set_ylabel("MOTA")
    ax1.set_title("(a) MOTA by Tracker Config\n(min_hits, max_age, dist_gate)")
    ax1.set_ylim(0.75, 0.85)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.axhline(y=0.8841, color="blue", linestyle=":", alpha=0.5, label="L2 upper bound")
    ax1.legend(fontsize=8)

    w = 0.25
    ax2.bar(x - w, fp_vals, w, label="FP", color="#FF9800")
    ax2.bar(x, fn_vals, w, label="FN", color="#2196F3")
    ax2.bar(x + w, idsw_vals, w, label="IDSW", color="#9C27B0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Error Components")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "fig12_tracker_params.png")
    plt.close()
    print("  ✓ fig12_tracker_params.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating figures to {OUT}/")
    fig1_training_curves()
    fig2_detection_comparison()
    fig3_parameter_efficiency()
    fig4_speed_accuracy()
    fig5_scalability()
    fig6_param_breakdown()
    fig7_three_level()
    fig8_field_prediction()
    fig9_trajectory()
    fig10_error_waterfall()
    fig11_timeline()
    fig12_tracker_params()
    print(f"\nDone! {len(list(OUT.glob('*.png')))} figures generated.")
