# src/report/make_bar_charts.py
"""
make_bar_charts.py
==================

Generate the four bar chart figures used in the paper from the per-model
**metrics** artifacts produced by the scoring pipeline.

What this script reads (per model under <results-root>/<model>/metrics/):
- <model>_scoring_v2_extended.json
    └─ Provides per-setting EM/F1 **percent** values (used for Figures 1–3).
- per_run_v2.csv
    └─ Provides request-level latencies (used for Figure 4).

What this script writes (into <out-dir>):
- bar_f1_by_setting.png     : F1 (%) across settings (Gold/Para/Dist/Both) by model
- bar_em_by_setting.png     : EM (%) across settings by model
- bar_delta_f1.png          : ΔF1 vs Gold (pp) for Para/Dist/Both by model
- bar_latency.png           : Latency p50 and p90 (ms) by model

Assumptions & conventions
-------------------------
- The JSON comes from `scoring_v2_extended.py` and has the schema:
      {
        "by_setting": {
          "gold": {"em_percent": float, "f1_percent": float, "n": int},
          "para": {...}, "dist": {...}, "para_dist": {...}
        },
        ...
      }
- The CSV `per_run_v2.csv` has a numeric `latency_ms` column.
- Settings are ordered as: Gold, Para, Dist, Para+Dist (labels are handled here).
- All EM/F1 values are **percent** (0–100), not fractions.

Quick start
-----------
$ python -m src.report.make_bar_charts \
    --results-root src/results \
    --out-dir src/reports/figures \
    --models gemini_pro gpt4o_mini gpt4o mistral7b llama31_8b

Optional flags
--------------
--sort-by {gold_f1,gold_em,none}   (default: gold_f1)
--no-value-labels                  (hide the numeric labels on bars)

If a model is missing a required file, a clear error is raised with the path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Defaults & display constants
# ----------------------------

DEFAULT_MODELS: List[str] = ["gemini_pro", "gpt4o_mini", "gpt4o", "mistral7b", "llama31_8b"]

# Internal order used for loading; labels are mapped below
SETTINGS: List[str] = ["gold", "para", "dist", "para_dist"]
SETTING_LABEL: Dict[str, str] = {"gold": "Gold", "para": "Para", "dist": "Dist", "para_dist": "Both"}

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ----------------------------
# IO helpers
# ----------------------------

def load_scores(results_root: str | Path, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load **percent** EM and F1 by setting for a single model.

    Parameters
    ----------
    results_root : str | Path
        Root folder containing per-model subfolders, e.g., "src/results".
    model : str
        Model folder name (e.g., "gpt4o", "llama31_8b").

    Returns
    -------
    (em, f1) : Tuple[np.ndarray, np.ndarray]
        Two arrays of shape (4,) ordered as [gold, para, dist, para_dist],
        containing **percent** values (floats). Missing settings are returned
        as NaN, which will display as empty bars.

    Notes
    -----
    - The function first looks for "<model>_scoring_v2_extended.json"; if not
      found, it falls back to "scoring_v2_extended.json".
    - Raises FileNotFoundError if neither exists.
    """
    metrics_dir = Path(results_root) / model / "metrics"
    fp = metrics_dir / f"{model}_scoring_v2_extended.json"
    if not fp.exists():
        # Backward compatibility for earlier file naming
        fp = metrics_dir / "scoring_v2_extended.json"
    if not fp.exists():
        raise FileNotFoundError(f"[{model}] Missing scoring JSON: {fp}")

    with fp.open("r", encoding="utf-8") as f:
        js = json.load(f)

    if "by_setting" not in js:
        raise KeyError(f"[{model}] 'by_setting' not found in: {fp}")

    by = js["by_setting"]
    em_vals: List[float] = []
    f1_vals: List[float] = []
    for s in SETTINGS:
        entry = by.get(s, None)
        if entry is None:
            em_vals.append(float("nan"))
            f1_vals.append(float("nan"))
            continue
        # pull percent; tolerate missing keys => NaN
        em_vals.append(float(entry.get("em_percent", float("nan"))))
        f1_vals.append(float(entry.get("f1_percent", float("nan"))))

    return np.array(em_vals, dtype=float), np.array(f1_vals, dtype=float)


def load_latency(results_root: str | Path, model: str) -> Tuple[float, float]:
    """
    Compute overall p50 and p90 latency (ms) for a model from `per_run_v2.csv`.

    Parameters
    ----------
    results_root : str | Path
        Root folder containing per-model subfolders, e.g., "src/results".
    model : str
        Model folder name.

    Returns
    -------
    (p50_ms, p90_ms) : Tuple[float, float]
        Median and 90th percentile latency in milliseconds.

    Raises
    ------
    FileNotFoundError
        If the per_run_v2.csv file does not exist.
    ValueError
        If the latency column is missing or non-numeric.
    """
    fp = Path(results_root) / model / "metrics" / "per_run_v2.csv"
    if not fp.exists():
        raise FileNotFoundError(f"[{model}] Missing per-run CSV: {fp}")

    df = pd.read_csv(fp)
    if "latency_ms" not in df.columns:
        raise ValueError(f"[{model}] Column 'latency_ms' not found in {fp}. Columns={list(df.columns)}")

    lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
    if lat.empty:
        return float("nan"), float("nan")

    p50 = float(lat.median())
    p90 = float(lat.quantile(0.90))
    return p50, p90


# ----------------------------
# Plotting helpers
# ----------------------------

def _format_model_name(m: str) -> str:
    """Humanize a snake_case model id for x-axis labels."""
    return m.replace("_", " ").title()


def bar_with_labels(
    ax: plt.Axes,
    x: np.ndarray,
    heights: Sequence[float],
    width: float = 0.2,
    label: str | None = None,
    offset: float = 0.0,
    show_values: bool = True,
    fmt: str = "{:.1f}",
    ypad: float = 0.4,
) -> List[plt.Rectangle]:
    """
    Draw a bar group and (optionally) annotate value labels above each bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : np.ndarray
        Baseline x positions (e.g., np.arange(n_models)).
    heights : Sequence[float]
        Bar heights; NaNs are allowed (matplotlib will render as empty bars).
    width : float, default 0.2
        Bar width.
    label : str | None, default None
        Legend label for this group.
    offset : float, default 0.0
        Horizontal offset to create grouped bars.
    show_values : bool, default True
        Whether to print the numeric values on top of bars.
    fmt : str, default "{:.1f}"
        Format string for value labels.
    ypad : float, default 0.4
        Vertical padding above bar tops for label placement.

    Returns
    -------
    bars : list[matplotlib.patches.Rectangle]
        The drawn bar artists.
    """
    bars = ax.bar(x + offset, heights, width=width, label=label)
    if show_values:
        for b, v in zip(bars, heights):
            try:
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + ypad,
                            fmt.format(v), ha="center", va="bottom", fontsize=9)
            except Exception:
                # If formatting fails (e.g., v is NaN or not a number), skip label
                pass
    return list(bars)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    """
    CLI entry point.

    Reads per-model metrics, builds four bar charts, and saves them to `--out-dir`.

    Exits with a clear error if any required input is missing.
    """
    ap = argparse.ArgumentParser(description="Build paper bar charts from metrics artifacts.")
    ap.add_argument("--results-root", default="src/results",
                    help="Root results directory containing <model>/metrics/ (default: src/results)")
    ap.add_argument("--out-dir", default="src/reports/figures",
                    help="Directory to write PNGs (default: src/reports/figures)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"Models to include (default: {' '.join(DEFAULT_MODELS)})")
    ap.add_argument("--sort-by", choices=["gold_f1", "gold_em", "none"], default="gold_f1",
                    help="Ordering for models in plots (default: gold_f1)")
    ap.add_argument("--no-value-labels", action="store_true",
                    help="Disable numeric labels above bars")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect data
    em_by: Dict[str, np.ndarray] = {}
    f1_by: Dict[str, np.ndarray] = {}
    lat_p50: Dict[str, float] = {}
    lat_p90: Dict[str, float] = {}

    for m in args.models:
        em, f1 = load_scores(args.results_root, m)
        em_by[m] = em
        f1_by[m] = f1
        p50, p90 = load_latency(args.results_root, m)
        lat_p50[m], lat_p90[m] = p50, p90

    # ---- sort models for display
    if args.sort_by == "gold_f1":
        key_fn = lambda m: (f1_by[m][0] if len(f1_by[m]) else float("-inf"))
        reverse = True
    elif args.sort_by == "gold_em":
        key_fn = lambda m: (em_by[m][0] if len(em_by[m]) else float("-inf"))
        reverse = True
    else:
        key_fn = None
        reverse = False

    models_sorted: List[str]
    if key_fn is None:
        models_sorted = list(args.models)
    else:
        models_sorted = sorted(args.models, key=key_fn, reverse=reverse)

    # Common x positions
    x = np.arange(len(models_sorted))
    w = 0.2
    show_vals = (not args.no_value_labels)

    # ---- 1) F1 by setting
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(SETTINGS):
        vals = [float(f1_by[m][i]) for m in models_sorted]
        bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i - 1.5) * w, show_values=show_vals)
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.set_title("F1 by Setting across Models")
    ax.legend()
    # Guard upper bound; ensure room for labels up to 100
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, max(100.0, ymax))
    fig.tight_layout()
    fig.savefig(out_dir / "bar_f1_by_setting.png", dpi=200)
    plt.close(fig)

    # ---- 2) EM by setting
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(SETTINGS):
        vals = [float(em_by[m][i]) for m in models_sorted]
        bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i - 1.5) * w, show_values=show_vals)
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("EM by Setting across Models")
    ax.legend()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, max(100.0, ymax))
    fig.tight_layout()
    fig.savefig(out_dir / "bar_em_by_setting.png", dpi=200)
    plt.close(fig)

    # ---- 3) ΔF1 vs Gold (pp) for Para, Dist, Both
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    for i, s in enumerate(["para", "dist", "para_dist"]):
        # delta_i = F1(setting) - F1(gold)
        idx = SETTINGS.index(s)
        delta = [float(f1_by[m][idx] - f1_by[m][0]) for m in models_sorted]
        bar_with_labels(ax, x, delta, width=w, label=SETTING_LABEL[s], offset=(i - 1.0) * w, show_values=show_vals)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("ΔF1 vs Gold (pp)")
    ax.set_title("Robustness: F1 Change vs Gold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_delta_f1.png", dpi=200)
    plt.close(fig)

    # ---- 4) Latency p50 and p90
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    p50_vals = [float(lat_p50[m]) for m in models_sorted]
    p90_vals = [float(lat_p90[m]) for m in models_sorted]
    bar_with_labels(ax, x, p50_vals, width=0.35, label="p50", offset=-0.18, show_values=show_vals)
    bar_with_labels(ax, x, p90_vals, width=0.35, label="p90", offset=+0.18, show_values=show_vals)
    ax.set_xticks(x)
    ax.set_xticklabels([_format_model_name(m) for m in models_sorted], rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bar_latency.png", dpi=200)
    plt.close(fig)

    print("Saved:")
    for fn in ["bar_f1_by_setting.png", "bar_em_by_setting.png", "bar_delta_f1.png", "bar_latency.png"]:
        print("  -", out_dir / fn)


if __name__ == "__main__":
    main()

    
# import argparse, json
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# MODELS = ["gemini_pro","gpt4o_mini","gpt4o","mistral7b","llama31_8b"]
# SETTINGS = ["gold","para","dist","para_dist"]
# SETTING_LABEL = {"gold":"Gold","para":"Para","dist":"Dist","para_dist":"Both"}

# def load_scores(results_root, model):
#     """Read EM/F1 (%) by setting from scoring_v2_extended.json."""
#     fp = Path(results_root)/model/"metrics"/f"{model}_scoring_v2_extended.json"
#     if not fp.exists():
#         fp = Path(results_root)/model/"metrics"/"scoring_v2_extended.json"
#     with open(fp, "r") as f:
#         js = json.load(f)
#     by = js["by_setting"]
#     em = [by[s]["em_percent"] for s in SETTINGS]
#     f1 = [by[s]["f1_percent"] for s in SETTINGS]
#     return np.array(em, float), np.array(f1, float)

# def load_latency(results_root, model):
#     """Read latency p50/p90 (ms) from per_run_v2.csv."""
#     fp = Path(results_root)/model/"metrics"/"per_run_v2.csv"
#     df = pd.read_csv(fp)
#     # If per_run is per question/run, we take overall p50/p90 across rows
#     p50 = float(df["latency_ms"].median())
#     p90 = float(df["latency_ms"].quantile(0.90))
#     return p50, p90

# def bar_with_labels(ax, x, heights, width=0.2, label=None, offset=0.0):
#     bars = ax.bar(x+offset, heights, width=width, label=label)
#     for b, v in zip(bars, heights):
#         ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, f"{v:.1f}",
#                 ha="center", va="bottom", fontsize=9)
#     return bars

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--results-root", default="src/results")
#     ap.add_argument("--out-dir", default="src/reports/figures")
#     args = ap.parse_args()

#     out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

#     # ---- collect data
#     em_by = {}
#     f1_by = {}
#     lat_p50 = {}
#     lat_p90 = {}
#     for m in MODELS:
#         em, f1 = load_scores(args.results_root, m)
#         em_by[m] = em
#         f1_by[m] = f1
#         p50, p90 = load_latency(args.results_root, m)
#         lat_p50[m], lat_p90[m] = p50, p90

#     # sort models by Gold F1 desc for cleaner ranking
#     models_sorted = sorted(MODELS, key=lambda m: f1_by[m][0], reverse=True)

#     # ---- 1) F1 by setting
#     x = np.arange(len(models_sorted))
#     w = 0.2
#     fig = plt.figure(figsize=(13,7))
#     ax = fig.add_subplot(111)
#     for i, s in enumerate(SETTINGS):
#         vals = [f1_by[m][i] for m in models_sorted]
#         bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i-1.5)*w)
#     ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
#     ax.set_ylabel("F1 (%)")
#     ax.set_title("F1 by Setting across Models")
#     ax.legend()
#     ax.set_ylim(0, max(100, ax.get_ylim()[1]))
#     fig.tight_layout()
#     fig.savefig(out/"bar_f1_by_setting.png", dpi=200)
#     plt.close(fig)

#     # ---- 2) EM by setting
#     fig = plt.figure(figsize=(13,7))
#     ax = fig.add_subplot(111)
#     for i, s in enumerate(SETTINGS):
#         vals = [em_by[m][i] for m in models_sorted]
#         bar_with_labels(ax, x, vals, width=w, label=SETTING_LABEL[s], offset=(i-1.5)*w)
#     ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
#     ax.set_ylabel("Exact Match (%)")
#     ax.set_title("EM by Setting across Models")
#     ax.legend()
#     ax.set_ylim(0, max(100, ax.get_ylim()[1]))
#     fig.tight_layout()
#     fig.savefig(out/"bar_em_by_setting.png", dpi=200)
#     plt.close(fig)

#     # ---- 3) ΔF1 vs Gold (pp) for Para, Dist, Both
#     fig = plt.figure(figsize=(13,7))
#     ax = fig.add_subplot(111)
#     for i, s in enumerate(["para","dist","para_dist"]):
#         delta = [f1_by[m][SETTINGS.index(s)] - f1_by[m][0] for m in models_sorted]
#         bar_with_labels(ax, x, delta, width=w, label=SETTING_LABEL[s], offset=(i-1)*w)
#     ax.axhline(0, linewidth=1)
#     ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
#     ax.set_ylabel("ΔF1 vs Gold (pp)")
#     ax.set_title("Robustness: F1 Change vs Gold")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(out/"bar_delta_f1.png", dpi=200)
#     plt.close(fig)

#     # ---- 4) Latency p50 and p90
#     fig = plt.figure(figsize=(13,7))
#     ax = fig.add_subplot(111)
#     p50_vals = [lat_p50[m] for m in models_sorted]
#     p90_vals = [lat_p90[m] for m in models_sorted]
#     bar_with_labels(ax, x, p50_vals, width=0.35, label="p50", offset=-0.18)
#     bar_with_labels(ax, x, p90_vals, width=0.35, label="p90", offset=+0.18)
#     ax.set_xticks(x); ax.set_xticklabels([m.replace("_"," ").title() for m in models_sorted], rotation=20, ha="right")
#     ax.set_ylabel("Latency (ms)")
#     ax.set_title("Latency by Model")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(out/"bar_latency.png", dpi=200)
#     plt.close(fig)

#     print("Saved:",
#           out/"bar_f1_by_setting.png",
#           out/"bar_em_by_setting.png",
#           out/"bar_delta_f1.png",
#           out/"bar_latency.png")

# if __name__ == "__main__":
#     main()