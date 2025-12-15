# Data Result: https://docs.google.com/document/d/153P6VBUJQupv6f73ZXZ3ZnAkzE6pNk6w7W0xcyYL4mY/edit?usp=sharing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ----------------------------
# Hardcoded benchmark results
# ----------------------------
cpu_names = ["Naive", "Separable", "Lee"]
cpu_no_o3_ms = np.array([405.898, 89.710, 56.377], dtype=float)

cpu_o3_ms = np.array([67.393, 5.355, 2.018], dtype=float)

gpu_names = [
    "naive",
    "naive+shared",
    "naive+separable",
    "naive+separable+shared",
    "lee+shared",
]
gpu_ms = np.array([0.054, 0.047, 0.069, 0.044, 0.047], dtype=float)

# Normalized versions (max in each subplot == 1.0)
cpu_no_o3_norm = cpu_no_o3_ms / cpu_no_o3_ms.max()
cpu_o3_norm = cpu_o3_ms / cpu_o3_ms.max()
gpu_norm = gpu_ms / gpu_ms.max()

# ----------------------------
# Color palettes (nice + consistent)
# ----------------------------
CPU_COLORS = ["#4C78A8", "#72B7B2", "#54A24B"]         # blue / teal / green
GPU_COLORS = ["#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]  # blue gradient


# ----------------------------
# Plot helpers (bars)
# ----------------------------
def barplot(ax, labels, values, title, ylabel,
            rotate_xticks=0, hide_xticklabels=False, ylim=None, colors=None):
    x = np.arange(len(labels))

    ax.bar(
        x, values,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.95,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=rotate_xticks,
        ha="right" if rotate_xticks else "center",
        rotation_mode="anchor",
    )

    if hide_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)

    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
    ax.margins(x=0.05)


# ----------------------------
# Roofline helper
# ----------------------------
def plot_roofline_fp32(
    out_pdf="dct_roofline_fp32.pdf",
    kernel_names=None,
    kernel_colors=None,
    kernel_ai=None,
    kernel_flops=None,
    ridge_ai=6.94,
    peak_fp32_flops=5_865_470_085_470.09
):
    assert kernel_ai is not None and kernel_flops is not None
    assert len(kernel_ai) == len(kernel_flops)
    n = len(kernel_ai)

    if kernel_names is None:
        kernel_names = [f"kernel{i}" for i in range(n)]
    if kernel_colors is None:
        kernel_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    # BW from ridge: BW = Peak / AI*
    bw_bytes_per_s = peak_fp32_flops / ridge_ai
    bw_gb_per_s = bw_bytes_per_s / 1e9
    peak_tflops = peak_fp32_flops / 1e12

    kernel_tflops = np.array(kernel_flops, dtype=float) / 1e12
    kernel_ai = np.array(kernel_ai, dtype=float)

    xmin = max(0.1, kernel_ai.min() / 5.0)
    xmax = max(200.0, kernel_ai.max() * 5.0)

    x_mem = np.logspace(np.log10(xmin), np.log10(ridge_ai), 200)
    y_mem = (bw_bytes_per_s * x_mem) / 1e12

    x_cmp = np.logspace(np.log10(ridge_ai), np.log10(xmax), 200)
    y_cmp = np.full_like(x_cmp, peak_tflops)

    fig = plt.figure(figsize=(10.5, 4.2), dpi=250, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    # Roof lines
    ax.plot(x_mem, y_mem, linewidth=2.6, color="#2c7fb8",
            label=f"HBM BW ≈ {bw_gb_per_s:.0f} GB/s", zorder=2)
    ax.plot(x_cmp, y_cmp, linewidth=2.6, color="#253494",
            label=f"FP32 peak ≈ {peak_tflops:.2f} TFLOP/s", zorder=2)

    # Ridge line
    ax.axvline(ridge_ai, linestyle="--", linewidth=1.2,
               color="#666666", alpha=0.7, zorder=1)
    ax.text(ridge_ai, peak_tflops * 0.12, f"AI*={ridge_ai:.2f}",
            rotation=90, va="bottom", ha="right",
            fontsize=9, color="#444444")

    # Kernel points: NO annotate, rely on legend only
    # Use different marker shapes too (optional but helps a lot)
    markers = ["o", "s", "^", "D", "P"]
    for i in range(n):
        ax.scatter(
            kernel_ai[i], kernel_tflops[i],
            s=95,
            marker=markers[i % len(markers)],
            color=kernel_colors[i],
            edgecolor="#111111",
            linewidth=0.9,
            zorder=5,
            label=kernel_names[i]
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title("Roofline (FP32) with 5 DCT Kernels", pad=10)

    y_min = max(0.05, kernel_tflops.min() / 3.0)
    y_max = max(peak_tflops * 1.2, kernel_tflops.max() * 2.0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)

    # Legend: single legend is enough now
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.get_legend().get_frame().set_alpha(0.92)
    ax.get_legend().get_frame().set_linewidth(0.6)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved roofline to: {out_pdf}")



# ----------------------------
# Figure layout: 2 rows x 3 cols using GridSpec
# ----------------------------
fig = plt.figure(figsize=(10, 5), dpi=200)
gs = gridspec.GridSpec(
    2, 3, figure=fig,
    width_ratios=[1, 1, 1.2],
    hspace=0.4, wspace=0.3
)

# Row 0: raw timings
ax_00 = fig.add_subplot(gs[0, 0])
barplot(ax_00, cpu_names, cpu_no_o3_ms, "CPU runtime (no -O3)", "Time (ms)",
        colors=CPU_COLORS)

ax_01 = fig.add_subplot(gs[0, 1])
barplot(ax_01, cpu_names, cpu_o3_ms, "CPU runtime (-O3)", "Time (ms)",
        colors=CPU_COLORS)

ax_02 = fig.add_subplot(gs[0, 2])
barplot(ax_02, gpu_names, gpu_ms, "GPU runtime", "Time (ms)",
        rotate_xticks=35, hide_xticklabels=True, colors=GPU_COLORS)

# Row 1: normalized timings
ax_10 = fig.add_subplot(gs[1, 0])
barplot(ax_10, cpu_names, cpu_no_o3_norm, "CPU normalized (no -O3)", "Normalized",
        ylim=(0, 1.05), colors=CPU_COLORS)

ax_11 = fig.add_subplot(gs[1, 1])
barplot(ax_11, cpu_names, cpu_o3_norm, "CPU normalized (-O3)", "Normalized",
        ylim=(0, 1.05), colors=CPU_COLORS)

ax_12 = fig.add_subplot(gs[1, 2])
barplot(ax_12, gpu_names, gpu_norm, "GPU normalized", "Normalized",
        rotate_xticks=35, ylim=(0, 1.05), colors=GPU_COLORS)

fig.subplots_adjust(bottom=0.18)

out_path = "dct_timings_grid.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")


# ----------------------------
# NEW: roofline plot (FP32 only) to another PDF
# ----------------------------
# Your 5 kernels roofline points:
# (Arithmetic Intensity FLOP/byte, Achieved FLOP/s)
roof_ai = np.array([27.28, 27.20, 2.89, 4.53, 1.46], dtype=float)
roof_flops = np.array([
    4_136_394_477_317.55,   # kernel 0
    4_858_267_181_467.18,   # kernel 1
    864_448_474_855.73,     # kernel 2
    871_634_247_714.05,     # kernel 3
    269_776_706_827.31,     # kernel 4
], dtype=float)

# FP32 roof params from Nsight (use ridge_ai + peak)
fp32_ridge_ai = 6.94
fp32_peak = 5_865_470_085_470.09

plot_roofline_fp32(
    out_pdf="dct_roofline_fp32.pdf",
    kernel_names=gpu_names,
    kernel_colors=GPU_COLORS,
    kernel_ai=roof_ai,
    kernel_flops=roof_flops,
    ridge_ai=fp32_ridge_ai,
    peak_fp32_flops=fp32_peak
)
