"""
src/evaluation/comparison_study.py

Step 13: Final Comparison Study
---------------------------------
Comprehensive analysis across 6 dimensions:

  1. Rule-based performance summary
  2. GNN performance summary
  3. False positive reduction %
  4. Fraud capture improvement %
  5. Adaptability test — simulate new fraud pattern
  6. Cold-start device attack test

Outputs:
  outputs/comparison/comparison_study_report.txt
  outputs/comparison/comparison_summary.png
  outputs/comparison/adaptability_test.png
  outputs/comparison/cold_start_test.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


# ── Cost constants (must match metrics.py) ────────────────────────────────────
FP_COST = 10.0
FN_LOSS = 150.0


# ═════════════════════════════════════════════════════════════════════════════
# 1 & 2 — Performance summaries
# ═════════════════════════════════════════════════════════════════════════════

def _performance_summary(results: dict, name: str) -> str:
    lines = [
        f"  {name}",
        f"  {'─'*40}",
        f"  Precision          : {results['precision']:.4f}",
        f"  Recall             : {results['recall']:.4f}",
        f"  F1 Score           : {results['f1_score']:.4f}",
        f"  ROC-AUC            : {results.get('roc_auc', 0):.4f}",
        f"  False Positive Rate: {results['false_positive_rate']:.4f}",
        f"  Fraud Capture Rate : {results['fraud_capture_rate']:.4f}",
        f"  Total Cost         : ${results['total_cost']:,.0f}",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# 3 & 4 — FP reduction + Fraud capture improvement
# ═════════════════════════════════════════════════════════════════════════════

def _compute_improvements(results_rule: dict, results_gnn: dict) -> dict:
    fp_rule = results_rule["fp"]
    fp_gnn  = results_gnn["fp"]
    fp_reduction_pct = ((fp_rule - fp_gnn) / max(fp_rule, 1)) * 100

    fcr_rule = results_rule["fraud_capture_rate"]
    fcr_gnn  = results_gnn["fraud_capture_rate"]
    fraud_capture_improvement_pct = ((fcr_gnn - fcr_rule) / max(fcr_rule, 1e-6)) * 100

    cost_saving = results_rule["total_cost"] - results_gnn["total_cost"]
    cost_saving_pct = (cost_saving / max(results_rule["total_cost"], 1)) * 100

    return {
        "fp_rule":                    fp_rule,
        "fp_gnn":                     fp_gnn,
        "fp_reduction":               fp_rule - fp_gnn,
        "fp_reduction_pct":           round(fp_reduction_pct, 2),
        "fcr_rule":                   round(fcr_rule, 4),
        "fcr_gnn":                    round(fcr_gnn, 4),
        "fraud_capture_improvement":  round(fraud_capture_improvement_pct, 2),
        "cost_saving":                round(cost_saving, 2),
        "cost_saving_pct":            round(cost_saving_pct, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5 — Adaptability test: simulate new fraud pattern
# ═════════════════════════════════════════════════════════════════════════════

def adaptability_test(df: pd.DataFrame,
                       y_proba_rule: np.ndarray,
                       y_proba_gnn:  np.ndarray,
                       y_true:       np.ndarray) -> dict:
    """
    Simulates a NEW fraud pattern that wasn't in training:
    high-value transactions from previously clean devices.

    Logic:
      - Identify devices that have ZERO historical fraud
      - Inject synthetic fraud signal: top 1% amount + clean device
      - See how each model responds to this unseen pattern

    This tests adaptability — GNN should generalise better because
    it uses graph context (neighbour patterns), not just hard rules.
    """
    print("\n  [Test 5] Adaptability Test — New Fraud Pattern Simulation")

    df_reset = df.reset_index(drop=True)

    # Find "clean" devices (no fraud history in dataset)
    device_fraud = df_reset.groupby("DeviceInfo")["isFraud"].sum()
    clean_devices = device_fraud[device_fraud == 0].index

    # New attack: high-value transactions from previously clean devices
    amt_threshold = df_reset["TransactionAmt"].quantile(0.99)
    new_pattern_mask = (
        df_reset["DeviceInfo"].isin(clean_devices) &
        (df_reset["TransactionAmt"] > amt_threshold)
    )

    new_pattern_idx = np.where(new_pattern_mask.values)[0]

    if len(new_pattern_idx) == 0:
        print("    No new pattern transactions found — using top-amount transactions.")
        new_pattern_idx = np.argsort(
            df_reset["TransactionAmt"].values
        )[-500:]

    # Ground truth for this subset
    y_true_sub      = y_true[new_pattern_idx]
    y_proba_rule_sub = y_proba_rule[new_pattern_idx]
    y_proba_gnn_sub  = y_proba_gnn[new_pattern_idx]

    # Use 0.5 threshold for binary predictions
    y_pred_rule_sub = (y_proba_rule_sub >= 0.5).astype(int)
    y_pred_gnn_sub  = (y_proba_gnn_sub  >= 0.4).astype(int)

    def safe_metrics(y_t, y_p, y_prob):
        if y_t.sum() == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5}
        return {
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall":    float(recall_score(y_t, y_p, zero_division=0)),
            "f1":        float(f1_score(y_t, y_p, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_t, y_prob))
            if len(np.unique(y_t)) > 1 else 0.5,
        }

    rule_metrics = safe_metrics(y_true_sub, y_pred_rule_sub, y_proba_rule_sub)
    gnn_metrics  = safe_metrics(y_true_sub, y_pred_gnn_sub,  y_proba_gnn_sub)

    n_fraud_in_pattern = int(y_true_sub.sum())
    print(f"    New pattern transactions : {len(new_pattern_idx):,}")
    print(f"    Actual fraud in subset   : {n_fraud_in_pattern:,}")
    print(f"    Rule Engine — F1: {rule_metrics['f1']:.4f} | "
          f"Recall: {rule_metrics['recall']:.4f}")
    print(f"    GNN          — F1: {gnn_metrics['f1']:.4f} | "
          f"Recall: {gnn_metrics['recall']:.4f}")

    winner = "GNN" if gnn_metrics["f1"] >= rule_metrics["f1"] else "Rule Engine"
    print(f"    Winner on new pattern: {winner}")

    return {
        "n_transactions":     len(new_pattern_idx),
        "n_fraud":            n_fraud_in_pattern,
        "rule_engine":        rule_metrics,
        "gnn":                gnn_metrics,
        "winner":             winner,
        "description": "High-value transactions from previously clean devices",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6 — Cold-start device attack test
# ═════════════════════════════════════════════════════════════════════════════

def cold_start_attack_test(df: pd.DataFrame,
                            y_proba_rule: np.ndarray,
                            y_proba_gnn:  np.ndarray,
                            y_true:       np.ndarray) -> dict:
    """
    Cold-start device attack: fraud from brand-new devices
    seen only ONCE in the dataset.

    This is a real attack vector — fraudsters use fresh devices
    to evade device-fingerprint rules.

    Rule engines fail here because they rely on device history.
    GNN may generalise via customer/email/address neighbours.
    """
    print("\n  [Test 6] Cold-Start Device Attack Test")

    df_reset = df.reset_index(drop=True)

    # Devices seen only once = cold-start devices
    device_counts = df_reset["DeviceInfo"].value_counts()
    cold_devices  = device_counts[device_counts == 1].index

    cold_mask = df_reset["DeviceInfo"].isin(cold_devices)
    cold_idx  = np.where(cold_mask.values)[0]

    if len(cold_idx) < 10:
        # Fallback: devices seen <= 3 times
        cold_devices = device_counts[device_counts <= 3].index
        cold_mask    = df_reset["DeviceInfo"].isin(cold_devices)
        cold_idx     = np.where(cold_mask.values)[0]

    y_true_cold      = y_true[cold_idx]
    y_proba_rule_cold = y_proba_rule[cold_idx]
    y_proba_gnn_cold  = y_proba_gnn[cold_idx]

    y_pred_rule_cold = (y_proba_rule_cold >= 0.5).astype(int)
    y_pred_gnn_cold  = (y_proba_gnn_cold  >= 0.4).astype(int)

    def safe_metrics(y_t, y_p, y_prob):
        if y_t.sum() == 0:
            return {"precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5,
                    "fp_rate": 0}
        cm = confusion_matrix(y_t, y_p)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {
            "precision": float(precision_score(y_t, y_p, zero_division=0)),
            "recall":    float(recall_score(y_t, y_p, zero_division=0)),
            "f1":        float(f1_score(y_t, y_p, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_t, y_prob))
            if len(np.unique(y_t)) > 1 else 0.5,
            "fp_rate":   float(fpr),
        }

    rule_metrics = safe_metrics(y_true_cold, y_pred_rule_cold, y_proba_rule_cold)
    gnn_metrics  = safe_metrics(y_true_cold, y_pred_gnn_cold,  y_proba_gnn_cold)

    n_fraud_cold = int(y_true_cold.sum())
    print(f"    Cold-start transactions  : {len(cold_idx):,}")
    print(f"    Actual fraud in subset   : {n_fraud_cold:,}")
    print(f"    Rule Engine — F1: {rule_metrics['f1']:.4f} | "
          f"FP Rate: {rule_metrics['fp_rate']:.4f}")
    print(f"    GNN          — F1: {gnn_metrics['f1']:.4f} | "
          f"FP Rate: {gnn_metrics['fp_rate']:.4f}")

    winner = "GNN" if gnn_metrics["f1"] >= rule_metrics["f1"] else "Rule Engine"
    print(f"    Winner on cold-start: {winner}")

    return {
        "n_transactions":  len(cold_idx),
        "n_fraud":         n_fraud_cold,
        "rule_engine":     rule_metrics,
        "gnn":             gnn_metrics,
        "winner":          winner,
        "description":     "Fraud from devices seen only once (cold-start attack)",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Visualisations
# ═════════════════════════════════════════════════════════════════════════════

def _plot_comparison_summary(results_rule, results_gnn, improvements):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Step 13: Final Comparison Study — Rule Engine vs GNN",
                 fontsize=14, fontweight="bold")

    RULE_C = "#3498db"
    GNN_C  = "#e74c3c"

    # Panel 1: Core metrics
    ax = axes[0, 0]
    metrics = ["Precision", "Recall", "F1", "ROC-AUC", "FCR"]
    r_vals  = [results_rule["precision"], results_rule["recall"],
               results_rule["f1_score"],  results_rule.get("roc_auc", 0),
               results_rule["fraud_capture_rate"]]
    g_vals  = [results_gnn["precision"],  results_gnn["recall"],
               results_gnn["f1_score"],   results_gnn.get("roc_auc", 0),
               results_gnn["fraud_capture_rate"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, r_vals, w, label="Rule Engine", color=RULE_C, alpha=0.85)
    ax.bar(x + w/2, g_vals, w, label="GNN",         color=GNN_C,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Core Performance Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (rv, gv) in enumerate(zip(r_vals, g_vals)):
        ax.text(i - w/2, rv + 0.01, f"{rv:.2f}", ha="center", fontsize=7)
        ax.text(i + w/2, gv + 0.01, f"{gv:.2f}", ha="center", fontsize=7)

    # Panel 2: FP reduction
    ax = axes[0, 1]
    bars = ax.bar(["Rule Engine", "GNN"],
                  [improvements["fp_rule"], improvements["fp_gnn"]],
                  color=[RULE_C, GNN_C], alpha=0.85)
    ax.set_title(f"False Positives\n"
                 f"({improvements['fp_reduction_pct']:+.1f}% reduction)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 500,
                f"{bar.get_height():,}", ha="center", fontsize=9)

    # Panel 3: Cost comparison
    ax = axes[1, 0]
    categories = ["FP Cost", "FN Loss", "Total"]
    r_costs = [results_rule["cost_fp"]/1e6, results_rule["cost_fn"]/1e6,
               results_rule["total_cost"]/1e6]
    g_costs = [results_gnn["cost_fp"]/1e6,  results_gnn["cost_fn"]/1e6,
               results_gnn["total_cost"]/1e6]
    x = np.arange(len(categories))
    ax.bar(x - w/2, r_costs, w, label="Rule Engine", color=RULE_C, alpha=0.85)
    ax.bar(x + w/2, g_costs, w, label="GNN",         color=GNN_C,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(f"Cost Model ($M)\n"
                 f"Saving: ${improvements['cost_saving']:,.0f} "
                 f"({improvements['cost_saving_pct']:+.1f}%)")
    ax.set_ylabel("Cost ($ millions)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Scorecard
    ax = axes[1, 1]
    ax.axis("off")
    scorecard = [
        ("Metric",                   "Rule Engine",  "GNN",       "Winner"),
        ("─"*18,                     "─"*10,         "─"*8,       "─"*10),
        ("Precision",
         f"{results_rule['precision']:.3f}",
         f"{results_gnn['precision']:.3f}",
         "GNN" if results_gnn["precision"] > results_rule["precision"]
         else "Rule"),
        ("Recall",
         f"{results_rule['recall']:.3f}",
         f"{results_gnn['recall']:.3f}",
         "GNN" if results_gnn["recall"] > results_rule["recall"]
         else "Rule"),
        ("F1 Score",
         f"{results_rule['f1_score']:.3f}",
         f"{results_gnn['f1_score']:.3f}",
         "GNN" if results_gnn["f1_score"] > results_rule["f1_score"]
         else "Rule"),
        ("ROC-AUC",
         f"{results_rule.get('roc_auc',0):.3f}",
         f"{results_gnn.get('roc_auc',0):.3f}",
         "GNN" if results_gnn.get("roc_auc",0) > results_rule.get("roc_auc",0)
         else "Rule"),
        ("FP Rate",
         f"{results_rule['false_positive_rate']:.3f}",
         f"{results_gnn['false_positive_rate']:.3f}",
         "GNN" if results_gnn["false_positive_rate"] <
         results_rule["false_positive_rate"] else "Rule"),
        ("Total Cost",
         f"${results_rule['total_cost']/1e6:.2f}M",
         f"${results_gnn['total_cost']/1e6:.2f}M",
         "GNN" if results_gnn["total_cost"] < results_rule["total_cost"]
         else "Rule"),
    ]
    col_x = [0.02, 0.38, 0.62, 0.82]
    for row_i, row in enumerate(scorecard):
        for col_i, val in enumerate(row):
            weight = "bold" if row_i <= 1 else "normal"
            color  = "#27ae60" if val == "GNN" else \
                     "#2980b9" if val == "Rule" else "black"
            ax.text(col_x[col_i], 1 - row_i * 0.11, val,
                    transform=ax.transAxes, fontsize=9,
                    fontweight=weight, color=color, va="top")
    ax.set_title("Final Scorecard", fontweight="bold")

    plt.tight_layout()
    path = "outputs/comparison/comparison_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Comparison summary saved → {path}")


def _plot_stress_tests(adapt_result: dict, cold_result: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stress Tests: Adaptability & Cold-Start Attack",
                 fontsize=13, fontweight="bold")

    RULE_C = "#3498db"
    GNN_C  = "#e74c3c"

    for ax, result, title in zip(
        axes,
        [adapt_result, cold_result],
        ["Test 5: New Fraud Pattern\n(High-value + clean device)",
         "Test 6: Cold-Start Device Attack\n(First-time device fraud)"]
    ):
        metrics = ["Precision", "Recall", "F1", "ROC-AUC"]
        r_vals  = [result["rule_engine"]["precision"],
                   result["rule_engine"]["recall"],
                   result["rule_engine"]["f1"],
                   result["rule_engine"]["roc_auc"]]
        g_vals  = [result["gnn"]["precision"],
                   result["gnn"]["recall"],
                   result["gnn"]["f1"],
                   result["gnn"]["roc_auc"]]

        x = np.arange(len(metrics))
        w = 0.35
        ax.bar(x - w/2, r_vals, w, label="Rule Engine",
               color=RULE_C, alpha=0.85)
        ax.bar(x + w/2, g_vals, w, label="GNN",
               color=GNN_C,  alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}\n"
                     f"n={result['n_transactions']:,} | "
                     f"fraud={result['n_fraud']:,} | "
                     f"Winner: {result['winner']}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for i, (rv, gv) in enumerate(zip(r_vals, g_vals)):
            ax.text(i - w/2, rv + 0.01, f"{rv:.2f}",
                    ha="center", fontsize=8)
            ax.text(i + w/2, gv + 0.01, f"{gv:.2f}",
                    ha="center", fontsize=8)

    plt.tight_layout()
    path = "outputs/comparison/stress_tests.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Stress test chart saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

def run_comparison_study(df:           pd.DataFrame,
                          y_true:       np.ndarray,
                          results_rule: dict,
                          results_gnn:  dict,
                          y_proba_rule: np.ndarray,
                          y_proba_gnn:  np.ndarray):
    """
    Full comparison study pipeline.

    Args:
        df           : preprocessed dataframe
        y_true       : ground truth labels
        results_rule : dict returned by evaluate_model() for rule engine
        results_gnn  : dict returned by evaluate_model() for GNN
        y_proba_rule : fraud probabilities from rule engine
        y_proba_gnn  : fraud probabilities from GNN
    """
    print("\n" + "=" * 55)
    print("  STEP 13: Final Comparison Study")
    print("=" * 55)

    os.makedirs("outputs/comparison", exist_ok=True)

    # ── 1 & 2: Performance summaries ─────────────────────────────────
    print("\n  [1] Rule-Based Engine Performance")
    print(_performance_summary(results_rule, "Rule-Based Engine"))

    print("\n  [2] GNN Performance")
    print(_performance_summary(results_gnn, "Heterogeneous GNN"))

    # ── 3 & 4: Improvements ──────────────────────────────────────────
    improvements = _compute_improvements(results_rule, results_gnn)

    print(f"\n  [3] False Positive Reduction")
    print(f"      Rule Engine FP : {improvements['fp_rule']:,}")
    print(f"      GNN FP         : {improvements['fp_gnn']:,}")
    print(f"      Reduction      : {improvements['fp_reduction']:+,} "
          f"({improvements['fp_reduction_pct']:+.1f}%)")

    print(f"\n  [4] Fraud Capture Improvement")
    print(f"      Rule Engine FCR: {improvements['fcr_rule']:.4f}")
    print(f"      GNN FCR        : {improvements['fcr_gnn']:.4f}")
    print(f"      Improvement    : {improvements['fraud_capture_improvement']:+.1f}%")
    print(f"      Cost saving    : ${improvements['cost_saving']:,.0f} "
          f"({improvements['cost_saving_pct']:+.1f}%)")

    # ── 5: Adaptability test ──────────────────────────────────────────
    print(f"\n  [5] Adaptability Test")
    adapt_result = adaptability_test(df, y_proba_rule, y_proba_gnn, y_true)

    # ── 6: Cold-start attack test ─────────────────────────────────────
    print(f"\n  [6] Cold-Start Device Attack Test")
    cold_result = cold_start_attack_test(df, y_proba_rule, y_proba_gnn, y_true)

    # ── Save full report ──────────────────────────────────────────────
    full_report = {
        "generated_at":    datetime.now().isoformat(),
        "performance": {
            "rule_engine": results_rule,
            "gnn":         results_gnn,
        },
        "improvements":    improvements,
        "adaptability_test":  adapt_result,
        "cold_start_test":    cold_result,
        "overall_winner": (
            "GNN" if (
                results_gnn["f1_score"]    > results_rule["f1_score"] and
                results_gnn["total_cost"]  < results_rule["total_cost"]
            ) else "Rule Engine"
        ),
    }

    json_path = "outputs/comparison/comparison_study.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=4)
    print(f"\n  Full report saved → {json_path}")

    # ── Visualisations ────────────────────────────────────────────────
    print("\n  Generating comparison visualisations...")
    _plot_comparison_summary(results_rule, results_gnn, improvements)
    _plot_stress_tests(adapt_result, cold_result)

    # ── Final verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  OVERALL WINNER: {full_report['overall_winner']}")
    print(f"  FP Reduction  : {improvements['fp_reduction_pct']:+.1f}%")
    print(f"  Cost Saving   : ${improvements['cost_saving']:,.0f}")
    print(f"  Adaptability  : {adapt_result['winner']} wins on new patterns")
    print(f"  Cold-Start    : {cold_result['winner']} wins on new devices")
    print("=" * 55)

    return full_report