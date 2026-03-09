"""
src/monitoring/drift_detection.py

Step 12: Drift Detection
-------------------------
Tracks fraud pattern evolution over time using two complementary methods:

  1. KL Divergence
       Measures how much the fraud score distribution has shifted
       between a reference window (training) and current window.
       KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

  2. Population Stability Index (PSI)
       Industry-standard metric used in credit risk and fraud systems.
       PSI < 0.10  → No significant shift  (stable)
       PSI < 0.25  → Moderate shift        (monitor)
       PSI >= 0.25 → Significant shift     (trigger retraining)

  3. Embedding Distribution Shift
       Tracks mean/std of GNN transaction embeddings over time
       to detect when the model's internal representations drift.

Output:
  outputs/monitoring/drift_report.txt
  outputs/monitoring/drift_score_distribution.png
  outputs/monitoring/psi_over_time.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy
from datetime import datetime


# ── Drift thresholds ──────────────────────────────────────────────────────────
PSI_STABLE   = 0.10
PSI_MONITOR  = 0.25   # above this → trigger retraining warning
KL_STABLE    = 0.05
KL_MONITOR   = 0.20


# ═════════════════════════════════════════════════════════════════════════════
# Core metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI).

    Compares the distribution of fraud scores between a reference
    population (training period) and a current population (new data).

    PSI = Σ (Current% - Reference%) * ln(Current% / Reference%)

    Args:
        reference : fraud probability scores from training window
        current   : fraud probability scores from current window
        n_bins    : number of buckets (10 is industry standard)

    Returns:
        PSI score (float)
    """
    # Build fixed bins from reference distribution
    bins      = np.linspace(0, 1, n_bins + 1)
    bins[0]   = -np.inf
    bins[-1]  = np.inf

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current,   bins=bins)[0]

    # Convert to proportions, avoid division by zero
    ref_pct = (ref_counts + 1e-6) / (len(reference) + 1e-6 * n_bins)
    cur_pct = (cur_counts + 1e-6) / (len(current)   + 1e-6 * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_kl_divergence(reference: np.ndarray, current: np.ndarray,
                           n_bins: int = 50) -> float:
    """
    KL Divergence between reference and current score distributions.

    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    Args:
        reference : reference fraud probability scores
        current   : current fraud probability scores
        n_bins    : histogram resolution

    Returns:
        KL divergence (float, >= 0; 0 = identical distributions)
    """
    bins = np.linspace(0, 1, n_bins + 1)

    p, _ = np.histogram(reference, bins=bins, density=True)
    q, _ = np.histogram(current,   bins=bins, density=True)

    # Smooth to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    return float(entropy(p, q))


def compute_embedding_drift(ref_embeddings: np.ndarray,
                              cur_embeddings: np.ndarray) -> dict:
    """
    Track mean and std shift in GNN transaction embeddings.

    A large shift in embedding space means the model is 'seeing'
    transactions that look very different from training data.

    Returns:
        dict with mean_shift, std_shift, max_dim_shift
    """
    ref_mean = ref_embeddings.mean(axis=0)
    cur_mean = cur_embeddings.mean(axis=0)
    ref_std  = ref_embeddings.std(axis=0)
    cur_std  = cur_embeddings.std(axis=0)

    mean_shift    = float(np.linalg.norm(cur_mean - ref_mean))
    std_shift     = float(np.linalg.norm(cur_std  - ref_std))
    max_dim_shift = float(np.abs(cur_mean - ref_mean).max())

    return {
        "mean_shift":     round(mean_shift,    4),
        "std_shift":      round(std_shift,     4),
        "max_dim_shift":  round(max_dim_shift, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Drift status helper
# ═════════════════════════════════════════════════════════════════════════════

def _psi_status(psi: float) -> tuple:
    if psi < PSI_STABLE:
        return "STABLE",  "✅"
    elif psi < PSI_MONITOR:
        return "MONITOR", "⚠️ "
    else:
        return "RETRAIN", "🔴"


def _kl_status(kl: float) -> tuple:
    if kl < KL_STABLE:
        return "STABLE",  "✅"
    elif kl < KL_MONITOR:
        return "MONITOR", "⚠️ "
    else:
        return "RETRAIN", "🔴"


# ═════════════════════════════════════════════════════════════════════════════
# Temporal drift simulation
# ═════════════════════════════════════════════════════════════════════════════

def simulate_temporal_drift(df: pd.DataFrame, y_proba: np.ndarray,
                              n_windows: int = 6) -> list:
    """
    Simulate drift detection over time by splitting data into
    chronological windows and computing PSI/KL for each.

    Uses TransactionDT as the time axis.

    Args:
        df       : preprocessed dataframe with TransactionDT
        y_proba  : fraud probabilities [N]
        n_windows: number of time windows to evaluate

    Returns:
        list of drift result dicts per window
    """
    df_reset    = df.reset_index(drop=True)
    time_col    = "TransactionDT"

    if time_col not in df_reset.columns:
        print("  TransactionDT not found — using row order as time proxy.")
        df_reset[time_col] = np.arange(len(df_reset))

    # Sort by time and assign window labels
    sorted_idx  = df_reset[time_col].argsort().values
    window_size = len(sorted_idx) // n_windows
    windows     = [
        sorted_idx[i * window_size: (i + 1) * window_size]
        for i in range(n_windows)
    ]

    # Reference = first window (training period)
    reference_scores = y_proba[windows[0]]

    results = []
    for i, window_idx in enumerate(windows):
        current_scores = y_proba[window_idx]

        psi = compute_psi(reference_scores, current_scores)
        kl  = compute_kl_divergence(reference_scores, current_scores)

        psi_stat, psi_icon = _psi_status(psi)
        kl_stat,  kl_icon  = _kl_status(kl)

        # Time range for this window
        time_min = df_reset.iloc[window_idx][time_col].min()
        time_max = df_reset.iloc[window_idx][time_col].max()

        result = {
            "window":       i + 1,
            "time_min":     int(time_min),
            "time_max":     int(time_max),
            "n_transactions": len(window_idx),
            "mean_fraud_score": round(float(current_scores.mean()), 4),
            "psi":          round(psi, 4),
            "psi_status":   psi_stat,
            "kl_divergence": round(kl, 4),
            "kl_status":    kl_stat,
            "action":       "RETRAIN" if (psi_stat == "RETRAIN" or
                                           kl_stat  == "RETRAIN") else
                            "MONITOR" if (psi_stat == "MONITOR" or
                                           kl_stat  == "MONITOR") else
                            "OK",
        }
        results.append(result)

        print(f"  Window {i+1}/{n_windows} "
              f"[{time_min:,} → {time_max:,}] "
              f"| PSI: {psi:.4f} {psi_icon} {psi_stat:<7} "
              f"| KL: {kl:.4f} {kl_icon} {kl_stat:<7} "
              f"| Action: {result['action']}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def _plot_score_distributions(df: pd.DataFrame, y_proba: np.ndarray,
                               n_windows: int = 3):
    """
    Overlay fraud score distributions from different time windows
    to visually show drift.
    """
    df_reset    = df.reset_index(drop=True)
    time_col    = "TransactionDT" if "TransactionDT" in df_reset.columns \
                  else df_reset.columns[0]
    sorted_idx  = df_reset[time_col].argsort().values
    window_size = len(sorted_idx) // n_windows

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Step 12: Fraud Score Distribution Drift Over Time",
                 fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e67e22", "#e74c3c"]
    labels = ["Early (reference)", "Mid period", "Recent (current)"]

    # Panel 1: Overlapping distributions
    ax = axes[0]
    for i in range(n_windows):
        idx    = sorted_idx[i * window_size: (i + 1) * window_size]
        scores = y_proba[idx]
        ax.hist(scores, bins=50, alpha=0.5, color=colors[i],
                label=labels[i], density=True)
    ax.set_xlabel("Fraud Probability Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Early vs Mid vs Recent")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Mean fraud score over time (rolling)
    ax = axes[1]
    chunk_size = max(1, len(y_proba) // 100)
    time_points, mean_scores = [], []
    for start in range(0, len(sorted_idx), chunk_size):
        chunk_idx = sorted_idx[start: start + chunk_size]
        time_points.append(start)
        mean_scores.append(float(y_proba[chunk_idx].mean()))

    ax.plot(time_points, mean_scores, color="#9b59b6", linewidth=1.5)
    ax.fill_between(time_points, mean_scores, alpha=0.2, color="#9b59b6")
    ax.axhline(np.mean(mean_scores), color="red", linestyle="--",
               alpha=0.6, label="Overall mean")
    ax.set_xlabel("Transaction index (chronological)")
    ax.set_ylabel("Mean Fraud Score")
    ax.set_title("Fraud Score Trend Over Time")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = "outputs/monitoring/drift_score_distribution.png"
    plt.savefig(path, dpi=150)
    print(f"  Score distribution plot saved → {path}")


def _plot_psi_over_time(drift_results: list):
    """Bar chart of PSI and KL across time windows."""
    windows = [f"W{r['window']}" for r in drift_results]
    psi_vals = [r["psi"]          for r in drift_results]
    kl_vals  = [r["kl_divergence"] for r in drift_results]

    # Colour bars by status
    psi_colors = []
    for r in drift_results:
        if r["psi_status"] == "STABLE":
            psi_colors.append("#2ecc71")
        elif r["psi_status"] == "MONITOR":
            psi_colors.append("#f39c12")
        else:
            psi_colors.append("#e74c3c")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Drift Detection — PSI & KL Divergence Over Time",
                 fontsize=13, fontweight="bold")

    # PSI
    ax = axes[0]
    bars = ax.bar(windows, psi_vals, color=psi_colors, alpha=0.85)
    ax.axhline(PSI_STABLE,  color="green",  linestyle="--",
               label=f"Stable threshold ({PSI_STABLE})")
    ax.axhline(PSI_MONITOR, color="orange", linestyle="--",
               label=f"Retrain threshold ({PSI_MONITOR})")
    ax.set_title("Population Stability Index (PSI)")
    ax.set_ylabel("PSI Score")
    ax.set_xlabel("Time Window")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, psi_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)

    # KL
    ax = axes[1]
    ax.plot(windows, kl_vals, marker="o", color="#3498db",
            linewidth=2, markersize=8)
    ax.fill_between(range(len(windows)), kl_vals, alpha=0.15, color="#3498db")
    ax.axhline(KL_STABLE,  color="green",  linestyle="--",
               label=f"Stable ({KL_STABLE})")
    ax.axhline(KL_MONITOR, color="orange", linestyle="--",
               label=f"Monitor ({KL_MONITOR})")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_title("KL Divergence Over Time")
    ax.set_ylabel("KL Divergence")
    ax.set_xlabel("Time Window")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for i, val in enumerate(kl_vals):
        ax.text(i, val + 0.002, f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = "outputs/monitoring/psi_over_time.png"
    plt.savefig(path, dpi=150)
    print(f"  PSI/KL chart saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_drift_detection(df: pd.DataFrame, y_proba: np.ndarray,
                         n_windows: int = 6):
    """
    Full drift detection pipeline.

    Args:
        df       : preprocessed dataframe
        y_proba  : fraud probabilities from GNN [N]
        n_windows: number of time windows to compare
    """
    print("\n" + "=" * 55)
    print("  STEP 12: Drift Detection")
    print("=" * 55)
    print(f"\n  Thresholds: PSI stable<{PSI_STABLE} | "
          f"monitor<{PSI_MONITOR} | retrain>={PSI_MONITOR}")
    print(f"              KL  stable<{KL_STABLE}  | "
          f"monitor<{KL_MONITOR}  | retrain>={KL_MONITOR}\n")

    os.makedirs("outputs/monitoring", exist_ok=True)

    # ── Temporal drift simulation ─────────────────────────────────────
    drift_results = simulate_temporal_drift(df, y_proba, n_windows)

    # ── Summary ───────────────────────────────────────────────────────
    retrain_windows = [r for r in drift_results if r["action"] == "RETRAIN"]
    monitor_windows = [r for r in drift_results if r["action"] == "MONITOR"]

    print(f"\n  Summary:")
    print(f"    Windows analysed  : {n_windows}")
    print(f"    Stable            : {n_windows - len(retrain_windows) - len(monitor_windows)}")
    print(f"    Monitor           : {len(monitor_windows)}")
    print(f"    Retrain triggered : {len(retrain_windows)}")

    if retrain_windows:
        print(f"\n  🔴 RETRAINING RECOMMENDED for windows: "
              f"{[r['window'] for r in retrain_windows]}")
    elif monitor_windows:
        print(f"\n  ⚠️  MONITORING ADVISED for windows: "
              f"{[r['window'] for r in monitor_windows]}")
    else:
        print(f"\n  ✅ Model distribution is STABLE across all windows.")

    # ── Save JSON report ──────────────────────────────────────────────
    report = {
        "generated_at":    datetime.now().isoformat(),
        "n_windows":       n_windows,
        "thresholds": {
            "psi_stable":  PSI_STABLE,
            "psi_monitor": PSI_MONITOR,
            "kl_stable":   KL_STABLE,
            "kl_monitor":  KL_MONITOR,
        },
        "windows":         drift_results,
        "recommendation":  "RETRAIN" if retrain_windows else
                           "MONITOR" if monitor_windows else "OK",
    }

    json_path = "outputs/monitoring/drift_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n  Drift report saved → {json_path}")

    # ── Visualisations ────────────────────────────────────────────────
    print("\n  Generating drift visualisations...")
    _plot_score_distributions(df, y_proba, n_windows=min(3, n_windows))
    _plot_psi_over_time(drift_results)

    print("\n  Drift detection complete.")
    return report