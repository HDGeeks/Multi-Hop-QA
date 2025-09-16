import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["gemini_pro","gpt4o_mini","gpt4o","mistral7b","llama31_8b"]
SETTINGS = ["gold","para","dist","para_dist"]
SETTING_LABEL = {"gold":"Gold","para":"Para","dist":"Dist","para_dist":"Both"}

def load_scores(results_root, model):
    """Read EM/F1 (%) by setting from scoring_v2_extended.json."""
    fp = Path(results_root)/model/"metrics"/f"{model}_scoring_v2_extended.json"
    if not fp.exists():
        fp = Path(results_root)/model/"metrics"/"scoring_v2_extended.json"
    with open(fp, "r") as f:
        js = json.load(f)
    by = js["by_setting"]
    em = [by[s]["em_percent"] for s in SETTINGS]
    f1 = [by[s]["f1_percent"] for s in SETTINGS]
    return np.array(em, float), np.array(f1, float)

def load_latency(results_root, model):
    """Read latency p50/p90 (ms) from per_run_v2.csv."""
    fp = Path(results_root)/model/"metrics"/"per_run_v2.csv"
    df = pd.read_csv(fp)
    # If per_run is per question/run, we take overall p50/p90 across rows
    p50 = float(df["latency_ms"].median())
    p90 = float(df["latency_ms"].quantile(0.90))
    return p50, p90

def bar_with_labels(ax, x, heights, width=0.2, label=None, offset=0.0):
    bars = ax.bar(x+offset, heights, width=width, label=label)
    for b, v in zip(bars, heights):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    return bars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="src/results_test")
    ap.add_argument("--out-dir", default="src/reports_test/figures")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- collect data
    em_by = {}
    f1_by = {}
    lat_p50 = {}
    lat_p90 = {}
    for m in MODELS:
        em, f1 = load_scores(args.results_root, m)
        em_by[m] = em
        f1_by[m] = f1
        p50, p90 = load_latency(args.results_root, m)
        lat_p50[m], lat_p90[m] = p50, p90

    # sort models by Gold F1 desc for cleaner ranking
    models_sorted = sorted(MODELS, key=lambda m: f1_by[m][0], reverse=True)

    # ---- 1) F1 by setting
    x = np.arange(len(models_sorted))
    w = 0.2
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(SETTINGS):
        vals = [f1_by[m][i] for m in models_sorted]
        bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i-1.5)*w)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_title("F1 by Setting across Models")
    ax.legend()
    ax.set_ylim(0, max(100, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out/"bar_f1_by_setting.png", dpi=200)
    plt.close(fig)

    # ---- 2) EM by setting
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(SETTINGS):
        vals = [em_by[m][i] for m in models_sorted]
        bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i-1.5)*w)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("EM by Setting across Models")
    ax.legend()
    ax.set_ylim(0, max(100, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(out/"bar_em_by_setting.png", dpi=200)
    plt.close(fig)

    # ---- 3) ΔF1 vs Gold (pp) for Para, Dist, Both
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(["para","dist","para_dist"]):
        delta = [f1_by[m][SETTINGS.index(s)] - f1_by[m][0] for m in models_sorted]
        bar_with_labels(ax, x, delta, width=w, label=SETTING_LABEL[s], offset=(i-1)*w)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("ΔF1 vs Gold (pp)")
    ax.set_title("Robustness: F1 Change vs Gold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"bar_delta_f1.png", dpi=200)
    plt.close(fig)

    # ---- 4) Latency p50 and p90
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111)
    p50_vals = [lat_p50[m] for m in models_sorted]
    p90_vals = [lat_p90[m] for m in models_sorted]
    bar_with_labels(ax, x, p50_vals, width=0.35, label="p50", offset=-0.18)
    bar_with_labels(ax, x, p90_vals, width=0.35, label="p90", offset=+0.18)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"bar_latency.png", dpi=200)
    plt.close(fig)

    print("Saved:",
          out/"bar_f1_by_setting.png",
          out/"bar_em_by_setting.png",
          out/"bar_delta_f1.png",
          out/"bar_latency.png")

if __name__ == "__main__":
    main()