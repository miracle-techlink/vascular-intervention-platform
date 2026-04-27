"""
从 eval_log.csv 生成论文图：
  Fig A: 收敛曲线（SR-holdout vs training steps）3组对比
  Fig B: 最终泛化对比柱状图（SR-train vs SR-holdout，3组）
  Fig C: 每个 held-out seed 的 per-seed SR（泛化细节）

用法：
  python examples/plot_ablation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODES   = ["single", "multi5", "multi10"]
LABELS  = ["Single (1 anatomy)", "Multi-5 (5 anatomies)", "Multi-10 (10 anatomies)"]
COLORS  = ["#e74c3c", "#3498db", "#2ecc71"]
HOLDOUT = [1000, 1001, 1002, 1003, 1004]
OUT_DIR = "models_ablation/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(mode):
    path = f"models_ablation/{mode}/eval_log.csv"
    if not os.path.exists(path):
        print(f"  [warn] {path} not found, skipping")
        return None
    return pd.read_csv(path)


# ── Fig A: 收敛曲线 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

for mode, label, color in zip(MODES, LABELS, COLORS):
    df = load_csv(mode)
    if df is None:
        continue
    steps = df["step"].values / 1000   # k-steps
    sr_ho = df["sr_holdout_mean"].values * 100
    # 平滑（3点移动平均）
    if len(sr_ho) >= 3:
        sr_smooth = np.convolve(sr_ho, np.ones(3)/3, mode="same")
        sr_smooth[[0, -1]] = sr_ho[[0, -1]]
    else:
        sr_smooth = sr_ho
    ax.plot(steps, sr_smooth, color=color, linewidth=2, label=label)
    ax.fill_between(steps, sr_smooth - 3, sr_smooth + 3, alpha=0.15, color=color)

ax.set_xlabel("Training Steps (×10³)")
ax.set_ylabel("Held-out Success Rate (%)")
ax.set_title("Generalization: Multi-Anatomy Training vs Single")
ax.set_ylim(0, 100)
ax.legend(loc="upper left", framealpha=0.9)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(ax.get_xlim()[-1] * 0.98, 52, "50% SR", ha="right", color="gray", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_convergence.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig_convergence.png", dpi=300, bbox_inches="tight")
print(f"  saved: {OUT_DIR}/fig_convergence.pdf")


# ── Fig B: 最终对比柱状图 ────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))

x      = np.arange(len(MODES))
width  = 0.35
train_vals, holdout_vals = [], []

for mode in MODES:
    df = load_csv(mode)
    if df is not None and len(df) > 0:
        train_vals.append(df["sr_train_mean"].iloc[-1] * 100)
        holdout_vals.append(df["sr_holdout_mean"].iloc[-1] * 100)
    else:
        train_vals.append(0.0)
        holdout_vals.append(0.0)

bars1 = ax2.bar(x - width/2, train_vals,  width, label="In-distribution",
                color=[c for c in COLORS], alpha=0.9, edgecolor="white")
bars2 = ax2.bar(x + width/2, holdout_vals, width, label="Held-out (generalization)",
                color=[c for c in COLORS], alpha=0.5, edgecolor="white", hatch="//")

ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, fontsize=10)
ax2.set_ylabel("Final Success Rate (%)")
ax2.set_ylim(0, 105)
ax2.set_title("In-distribution vs Held-out SR at 200k Steps")

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    if h > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., h + 1,
                 f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

patch1 = mpatches.Patch(color="gray",    alpha=0.9, label="In-distribution")
patch2 = mpatches.Patch(color="gray",    alpha=0.5, hatch="//", label="Held-out")
ax2.legend(handles=[patch1, patch2], loc="upper left")
plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig_final_sr.pdf", dpi=300, bbox_inches="tight")
fig2.savefig(f"{OUT_DIR}/fig_final_sr.png", dpi=300, bbox_inches="tight")
print(f"  saved: {OUT_DIR}/fig_final_sr.pdf")


# ── Fig C: Per-seed held-out SR（multi10 最终结果）───────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 3.5))

for mode, label, color in zip(MODES, LABELS, COLORS):
    df = load_csv(mode)
    if df is None or len(df) == 0:
        continue
    last = df.iloc[-1]
    per_seed = [last.get(f"sr_holdout_s{s}", 0.0) * 100 for s in HOLDOUT]
    ax3.plot(range(len(HOLDOUT)), per_seed, "o-", color=color,
             linewidth=1.5, markersize=6, label=label)

ax3.set_xticks(range(len(HOLDOUT)))
ax3.set_xticklabels([f"Anat-{i+1}" for i in range(len(HOLDOUT))])
ax3.set_ylabel("Success Rate (%)")
ax3.set_ylim(0, 105)
ax3.set_title("Per-Anatomy Held-out SR (Final Model)")
ax3.legend(loc="lower right")
ax3.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig_per_seed.pdf", dpi=300, bbox_inches="tight")
fig3.savefig(f"{OUT_DIR}/fig_per_seed.png", dpi=300, bbox_inches="tight")
print(f"  saved: {OUT_DIR}/fig_per_seed.pdf")
print("  全部图表生成完毕。")
