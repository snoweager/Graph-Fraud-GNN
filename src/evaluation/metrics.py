"""
src/evaluation/metrics.py

Step 10: Full Evaluation Framework
------------------------------------
Metrics:
  - Accuracy, Precision, Recall, F1
  - ROC-AUC
  - False Positive Rate (FPR)
  - Fraud Capture Rate (same as Recall, surfaced explicitly)
  - Cost-based metric: (FP_cost * FP) + (FN_loss * FN)

Also provides:
  - compare_models(): side-by-side GNN vs Rule Engine
  - Saves comparison chart to outputs/evaluation/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ── Cost assumptions (tunable) ────────────────────────────────────────────────
# FP: legitimate transaction wrongly blocked → customer friction, ~$10 avg cost
# FN: fraud missed → full transaction loss,  ~$150 avg loss
FP_COST  = 10.0
FN_LOSS  = 150.0


# ═════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model") -> dict:
    """
    Evaluate a single model and print full report.

    Parameters:
        y_true      : Ground truth labels (0/1)
        y_pred      : Binary predictions  (0/1)
        y_proba     : Fraud probabilities (float) — needed for ROC-AUC
        model_name  : Display name

    Returns:
        dict of all metrics (JSON-serialisable)
    """

    print("\n" + "=" * 50)
    print(f"{model_name} Evaluation")
    print("=" * 50)

    # ── Standard metrics ──────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # ── ROC-AUC ───────────────────────────────────────────────────────
    roc = None
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_true, y_proba)
            print(f"ROC-AUC   : {roc:.4f}")
        except Exception:
            print("ROC-AUC   : Could not compute")

    # ── Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)

    # ── Extended metrics ──────────────────────────────────────────────

    # False Positive Rate: how often do we block a legit transaction?
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Fraud Capture Rate = Recall (explicit label for business audience)
    fraud_capture_rate = rec

    # Cost metric
    total_cost = (FP_COST * fp) + (FN_LOSS * fn)

    print(f"\nFalse Positive Rate : {fpr:.4f}  ({fp:,} legit txns wrongly blocked)")
    print(f"Fraud Capture Rate  : {fraud_capture_rate:.4f}  ({tp:,} of {tp+fn:,} frauds caught)")
    print(f"\nCost Model (FP=${FP_COST}, FN=${FN_LOSS}):")
    print(f"  FP cost : ${FP_COST * fp:>12,.0f}  ({fp:,} false positives)")
    print(f"  FN loss : ${FN_LOSS * fn:>12,.0f}  ({fn:,} missed frauds)")
    print(f"  TOTAL   : ${total_cost:>12,.0f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("=" * 50)

    os.makedirs("outputs/metrics", exist_ok=True)

    return {
        "model":              model_name,
        "accuracy":           float(acc),
        "precision":          float(prec),
        "recall":             float(rec),
        "f1_score":           float(f1),
        "roc_auc":            float(roc) if roc is not None else None,
        "false_positive_rate": float(fpr),
        "fraud_capture_rate": float(fraud_capture_rate),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "cost_fp":            float(FP_COST * fp),
        "cost_fn":            float(FN_LOSS * fn),
        "total_cost":         float(total_cost),
        "confusion_matrix":   cm.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Side-by-side comparison
# ═════════════════════════════════════════════════════════════════════════════

def compare_models(
    y_true,
    results_rule:  dict,
    results_gnn:   dict,
    y_proba_rule:  np.ndarray = None,
    y_proba_gnn:   np.ndarray = None,
):
    """
    Print and visualise a side-by-side comparison of Rule Engine vs GNN.
    Saves to outputs/evaluation/model_comparison.png
    """

    os.makedirs("outputs/evaluation", exist_ok=True)

    # ── Console table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Rule-Based Engine vs Heterogeneous GNN")
    print("=" * 65)

    metrics = [
        ("Precision",           "precision"),
        ("Recall",              "recall"),
        ("F1 Score",            "f1_score"),
        ("ROC-AUC",             "roc_auc"),
        ("False Positive Rate", "false_positive_rate"),
        ("Fraud Capture Rate",  "fraud_capture_rate"),
    ]

    print(f"  {'Metric':<25} {'Rule Engine':>12} {'GNN':>12}  {'Winner':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*8}")

    for label, key in metrics:
        rv = results_rule.get(key)
        gv = results_gnn.get(key)
        if rv is None or gv is None:
            continue
        # Lower FPR is better; higher is better for everything else
        if key == "false_positive_rate":
            winner = "Rule ✓" if rv < gv else "GNN  ✓"
        else:
            winner = "Rule ✓" if rv > gv else "GNN  ✓"
        print(f"  {label:<25} {rv:>12.4f} {gv:>12.4f}  {winner:>8}")

    print()
    print(f"  {'Cost (FP+FN)':<25} "
          f"${results_rule['total_cost']:>11,.0f} "
          f"${results_gnn['total_cost']:>11,.0f}  "
          f"{'Rule ✓' if results_rule['total_cost'] < results_gnn['total_cost'] else 'GNN  ✓':>8}")

    fp_reduction = results_rule["fp"] - results_gnn["fp"]
    cost_saving  = results_rule["total_cost"] - results_gnn["total_cost"]
    print(f"\n  False Positive reduction : {fp_reduction:+,} transactions")
    print(f"  Estimated cost saving    : ${cost_saving:+,.0f}")
    print("=" * 65)

    # Save comparison JSON
    comparison = {
        "rule_engine": results_rule,
        "gnn":         results_gnn,
        "fp_reduction":  fp_reduction,
        "cost_saving":   cost_saving,
    }
    with open("outputs/metrics/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    # ── Visualisation ─────────────────────────────────────────────────
    _plot_comparison(y_true, results_rule, results_gnn,
                     y_proba_rule, y_proba_gnn)


def _plot_comparison(y_true, r_rule, r_gnn, y_proba_rule, y_proba_gnn):
    """Generate 4-panel comparison chart."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Rule-Based Engine vs Heterogeneous GNN — Evaluation Framework",
                 fontsize=14, fontweight="bold")

    RULE_COLOR = "#3498db"
    GNN_COLOR  = "#e74c3c"

    # ── Panel 1: Bar chart of key metrics ────────────────────────────
    ax = axes[0, 0]
    metric_labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    rule_vals = [r_rule["precision"], r_rule["recall"],
                 r_rule["f1_score"],  r_rule.get("roc_auc") or 0]
    gnn_vals  = [r_gnn["precision"],  r_gnn["recall"],
                 r_gnn["f1_score"],   r_gnn.get("roc_auc") or 0]

    x = np.arange(len(metric_labels))
    w = 0.35
    ax.bar(x - w/2, rule_vals, w, label="Rule Engine", color=RULE_COLOR, alpha=0.85)
    ax.bar(x + w/2, gnn_vals,  w, label="GNN",         color=GNN_COLOR,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Key Metrics Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (rv, gv) in enumerate(zip(rule_vals, gnn_vals)):
        ax.text(i - w/2, rv + 0.01, f"{rv:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, gv + 0.01, f"{gv:.2f}", ha="center", fontsize=8)

    # ── Panel 2: Cost comparison ──────────────────────────────────────
    ax = axes[0, 1]
    categories  = ["FP Cost", "FN Loss", "Total Cost"]
    rule_costs  = [r_rule["cost_fp"], r_rule["cost_fn"], r_rule["total_cost"]]
    gnn_costs   = [r_gnn["cost_fp"],  r_gnn["cost_fn"],  r_gnn["total_cost"]]

    x = np.arange(len(categories))
    ax.bar(x - w/2, [c/1e6 for c in rule_costs], w,
           label="Rule Engine", color=RULE_COLOR, alpha=0.85)
    ax.bar(x + w/2, [c/1e6 for c in gnn_costs],  w,
           label="GNN",         color=GNN_COLOR,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Cost Model Comparison")
    ax.set_ylabel("Cost ($ millions)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: ROC curves ───────────────────────────────────────────
    ax = axes[1, 0]
    if y_proba_rule is not None:
        fpr_r, tpr_r, _ = roc_curve(y_true, y_proba_rule)
        ax.plot(fpr_r, tpr_r, color=RULE_COLOR, lw=2,
                label=f"Rule Engine (AUC={r_rule.get('roc_auc', 0):.3f})")
    if y_proba_gnn is not None:
        fpr_g, tpr_g, _ = roc_curve(y_true, y_proba_gnn)
        ax.plot(fpr_g, tpr_g, color=GNN_COLOR, lw=2,
                label=f"GNN (AUC={r_gnn.get('roc_auc', 0):.3f})")
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 4: FP / FN breakdown ────────────────────────────────────
    ax = axes[1, 1]
    labels   = ["True Neg", "False Pos\n(wrongly blocked)", "False Neg\n(missed fraud)", "True Pos"]
    rule_cm  = [r_rule["tn"], r_rule["fp"], r_rule["fn"], r_rule["tp"]]
    gnn_cm   = [r_gnn["tn"],  r_gnn["fp"],  r_gnn["fn"],  r_gnn["tp"]]
    colors   = ["#2ecc71", "#e67e22", "#e74c3c", "#27ae60"]

    x = np.arange(len(labels))
    ax.bar(x - w/2, [v/1000 for v in rule_cm], w,
           label="Rule Engine", color=RULE_COLOR, alpha=0.85)
    ax.bar(x + w/2, [v/1000 for v in gnn_cm],  w,
           label="GNN",         color=GNN_COLOR,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Confusion Matrix Breakdown")
    ax.set_ylabel("Count (thousands)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "outputs/evaluation/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Comparison chart saved → {path}")
