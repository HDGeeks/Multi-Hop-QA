import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# DATA (from your tables)
# --------------------------
models = ["Gemini Pro", "GPT-4o Mini", "GPT-4o", "Mistral-7B", "LLaMA-3.1-8B"]

# F1 per setting
f1_gold  = [99.0, 95.0, 94.3, 88.6, 58.4]
f1_para  = [94.3, 92.2, 90.3, 86.4, 55.3]
f1_dist  = [99.0, 95.0, 92.6, 88.9, 57.6]
f1_both  = [95.0, 92.5, 89.8, 85.5, 55.4]

# Latency (p50 ms)
latency = [615.0, 706.0, 716.0, 778.5, 818.0]

# Efficiency = Gold F1 / (latency in sec)
efficiency = [99.0/(615/1000), 95.0/(706/1000), 94.3/(716/1000),
              88.6/(778.5/1000), 58.4/(818/1000)]

# --------------------------
# Helpers
# --------------------------
def annotate_bars(ax, bars, fmt="{:.1f}", ypad=2, rotation=0):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2, h + ypad,
            fmt.format(h),
            ha="center", va="bottom", fontsize=8, rotation=rotation
        )

# =======================
# FIGURE 1: F1 by Setting
# =======================
x = np.arange(len(models))
w = 0.2

plt.figure(figsize=(10,6))
b1 = plt.bar(x - 1.5*w, f1_gold,  width=w, label="Gold")
b2 = plt.bar(x - 0.5*w, f1_para,  width=w, label="Para")
b3 = plt.bar(x + 0.5*w, f1_dist,  width=w, label="Dist")
b4 = plt.bar(x + 1.5*w, f1_both,  width=w, label="Both")

# Labels above bars (percent)
annotate_bars(plt.gca(), b1, fmt="{:.1f}")
annotate_bars(plt.gca(), b2, fmt="{:.1f}")
annotate_bars(plt.gca(), b3, fmt="{:.1f}")
annotate_bars(plt.gca(), b4, fmt="{:.1f}")

plt.xticks(x, models, rotation=30, ha="right")
plt.ylabel("F1 (%)")
plt.title("F1 by Setting across Models")
plt.ylim(0, 110)  # headroom for labels
plt.legend()
plt.tight_layout()
plt.savefig("f1_by_setting.png", dpi=300, bbox_inches="tight")
plt.close()

# =================================
# FIGURE 2: Latency & Efficiency
# =================================
fig, ax1 = plt.subplots(figsize=(10,6))
ax2 = ax1.twinx()

b_lat = ax1.bar(x - 0.2, latency,    width=0.4, alpha=0.8, label="Latency (ms)")
b_eff = ax2.bar(x + 0.2, efficiency, width=0.4, alpha=0.8, label="Efficiency (F1/s)")

# Labels above bars (no decimals for ms, 1 decimal for Efficiency)
annotate_bars(ax1, b_lat, fmt="{:.0f}", ypad=8)
annotate_bars(ax2, b_eff, fmt="{:.1f}", ypad=8)

ax1.set_ylabel("Latency (ms)")
ax2.set_ylabel("Efficiency (F1/s)")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30, ha="right")
ax1.set_ylim(0, max(latency)*1.25)
ax2.set_ylim(0, max(efficiency)*1.25)

# One legend for both
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2,
           loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2)

plt.title("Latency vs Efficiency")
plt.tight_layout()
plt.savefig("latency_efficiency.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Saved f1_by_setting.png and latency_efficiency.png with value labels.")