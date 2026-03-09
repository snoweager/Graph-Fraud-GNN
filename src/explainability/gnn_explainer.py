"""
src/explainability/gnn_explainer.py

Step 11: Explainability Layer
------------------------------
Implements two complementary explainability approaches:

  Option A — GNNExplainer
      Uses torch_geometric's Explainer API to find the minimal
      subgraph + edge mask that explains a fraud prediction.
      Outputs: fraud_subgraph_tx{id}.png per transaction

  Option B — Compliance Reasoning Engine
      Builds human-readable explanations from graph structure:
        "Transaction flagged because:
          • Device linked to 4 risky accounts
          • Customer has 3 prior fraud transactions
          • Email domain linked to 12 suspicious transactions"
      Outputs: fraud_explanation_report.txt + explanation cards

Run from main.py:
    from src.explainability.gnn_explainer import run_explainability
    run_explainability(model, graph, df_processed, model_kwargs)
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════
# Option B — Compliance Reasoning Engine
# (built first because it works regardless of PyG explainer version)
# ═════════════════════════════════════════════════════════════════════

class ComplianceReasoningEngine:
    """
    Produces human-readable fraud explanations from graph topology
    and model scores — no additional ML required.

    For each flagged transaction, it inspects:
      - Customer history (fraud rate, transaction volume)
      - Device risk    (how many fraud txns used this device)
      - Email risk     (fraud rate on this email domain)
      - Address risk   (fraud rate on this billing address)
      - Transaction amount vs customer average
    """

    # Risk thresholds
    DEVICE_RISK_COUNT    = 3    # device linked to >= N fraud txns → risky
    EMAIL_RISK_RATE      = 0.10 # email domain fraud rate >= 10%   → risky
    ADDRESS_RISK_RATE    = 0.10
    CUSTOMER_RISK_RATE   = 0.15 # customer historical fraud rate   → risky
    AMOUNT_SPIKE_FACTOR  = 3.0  # amount > N × customer avg        → spike

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: preprocessed dataframe (must contain isFraud, card1,
                DeviceInfo, P_emaildomain, addr1, TransactionAmt)
        """
        self.df = df
        self._build_risk_indexes()

    def _build_risk_indexes(self):
        """Pre-compute per-entity risk statistics from the full dataset."""
        df = self.df

        # Customer stats
        cust = df.groupby("card1").agg(
            tx_count   =("TransactionID", "count"),
            fraud_count=("isFraud", "sum"),
            avg_amt    =("TransactionAmt", "mean"),
        )
        cust["fraud_rate"] = cust["fraud_count"] / cust["tx_count"]
        self.customer_stats = cust

        # Device stats
        dev = df.groupby("DeviceInfo").agg(
            tx_count   =("TransactionID", "count"),
            fraud_count=("isFraud", "sum"),
        )
        dev["fraud_rate"] = dev["fraud_count"] / dev["tx_count"]
        self.device_stats = dev

        # Email stats
        email = df.groupby("P_emaildomain").agg(
            tx_count   =("TransactionID", "count"),
            fraud_count=("isFraud", "sum"),
        )
        email["fraud_rate"] = email["fraud_count"] / email["tx_count"]
        self.email_stats = email

        # Address stats
        addr = df.groupby("addr1").agg(
            tx_count   =("TransactionID", "count"),
            fraud_count=("isFraud", "sum"),
        )
        addr["fraud_rate"] = addr["fraud_count"] / addr["tx_count"]
        self.address_stats = addr

    def explain(self, row: pd.Series, fraud_score: float) -> dict:
        """
        Generate a compliance-grade explanation for one transaction.

        Returns:
            dict with 'reasons' list, 'risk_factors' dict, 'verdict'
        """
        reasons      = []
        risk_factors = {}

        customer = row.get("card1",         "unknown")
        device   = row.get("DeviceInfo",    "unknown")
        email    = row.get("P_emaildomain", "unknown")
        address  = row.get("addr1",         "unknown")
        amount   = row.get("TransactionAmt", 0.0)

        # ── Customer history ──────────────────────────────────────────
        if customer in self.customer_stats.index:
            cs = self.customer_stats.loc[customer]
            risk_factors["customer_fraud_rate"] = round(float(cs["fraud_rate"]), 4)
            risk_factors["customer_tx_count"]   = int(cs["tx_count"])
            risk_factors["customer_avg_amt"]     = round(float(cs["avg_amt"]), 2)

            if cs["fraud_rate"] >= self.CUSTOMER_RISK_RATE:
                reasons.append(
                    f"Customer has a {cs['fraud_rate']*100:.1f}% historical "
                    f"fraud rate ({int(cs['fraud_count'])} fraud txns)"
                )
            if cs["avg_amt"] > 0 and amount > self.AMOUNT_SPIKE_FACTOR * cs["avg_amt"]:
                spike = amount / cs["avg_amt"]
                reasons.append(
                    f"Transaction amount ${amount:.2f} is {spike:.1f}× "
                    f"the customer's average (${cs['avg_amt']:.2f})"
                )

        # ── Device risk ───────────────────────────────────────────────
        if device in self.device_stats.index:
            ds = self.device_stats.loc[device]
            risk_factors["device_fraud_count"] = int(ds["fraud_count"])
            risk_factors["device_tx_count"]    = int(ds["tx_count"])

            if ds["fraud_count"] >= self.DEVICE_RISK_COUNT:
                reasons.append(
                    f"Device '{device}' linked to "
                    f"{int(ds['fraud_count'])} prior fraud transactions"
                )

        # ── Email domain risk ─────────────────────────────────────────
        if email in self.email_stats.index:
            es = self.email_stats.loc[email]
            risk_factors["email_fraud_rate"] = round(float(es["fraud_rate"]), 4)

            if es["fraud_rate"] >= self.EMAIL_RISK_RATE:
                reasons.append(
                    f"Email domain '{email}' has a "
                    f"{es['fraud_rate']*100:.1f}% fraud rate"
                )

        # ── Address risk ──────────────────────────────────────────────
        if address in self.address_stats.index:
            ads = self.address_stats.loc[address]
            risk_factors["address_fraud_rate"] = round(float(ads["fraud_rate"]), 4)

            if ads["fraud_rate"] >= self.ADDRESS_RISK_RATE:
                reasons.append(
                    f"Billing address area '{address}' has a "
                    f"{ads['fraud_rate']*100:.1f}% fraud rate"
                )

        # ── Verdict ───────────────────────────────────────────────────
        if fraud_score >= 0.70:
            verdict = "HIGH RISK — Recommend BLOCK"
        elif fraud_score >= 0.40:
            verdict = "MEDIUM RISK — Recommend REVIEW"
        else:
            verdict = "LOW RISK — PASS"

        if not reasons:
            reasons.append(
                "Model detected unusual pattern in transaction embedding "
                "(no single dominant rule trigger)"
            )

        return {
            "fraud_score":  round(fraud_score, 4),
            "verdict":      verdict,
            "reasons":      reasons,
            "risk_factors": risk_factors,
        }

    def format_report(self, tx_id, explanation: dict) -> str:
        """Format explanation as a readable compliance report block."""
        lines = [
            f"{'─'*55}",
            f"  Transaction : {tx_id}",
            f"  Fraud Score : {explanation['fraud_score']:.4f}",
            f"  Verdict     : {explanation['verdict']}",
            f"",
            f"  Why flagged:",
        ]
        for i, reason in enumerate(explanation["reasons"], 1):
            lines.append(f"    {i}. {reason}")
        lines.append(f"{'─'*55}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Option A — GNNExplainer wrapper
# ═════════════════════════════════════════════════════════════════════

def run_gnn_explainer(model, graph, target_tx_indices: list,
                       device: torch.device):
    """
    Run torch_geometric's Explainer on a set of transactions.
    Gracefully falls back to a message if the PyG version doesn't
    support HeteroData explanation.

    Args:
        model:              trained HeteroFraudGNN
        graph:              HeteroData graph with features
        target_tx_indices:  list of transaction node indices to explain
        device:             torch device
    """
    try:
        from torch_geometric.explain import Explainer, GNNExplainer

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="probs",
            ),
        )

        os.makedirs("outputs/explainability", exist_ok=True)
        results = []

        for tx_idx in target_tx_indices:
            try:
                explanation = explainer(
                    x=graph["transaction"].x,
                    edge_index=graph["customer", "makes", "transaction"].edge_index,
                    index=tx_idx,
                )
                results.append({
                    "tx_index":  tx_idx,
                    "node_mask": explanation.node_mask.cpu().numpy(),
                    "edge_mask": explanation.edge_mask.cpu().numpy(),
                })
                print(f"  GNNExplainer: explained TX#{tx_idx}")
            except Exception as e:
                print(f"  GNNExplainer: skipped TX#{tx_idx} — {e}")

        return results

    except ImportError:
        print("  GNNExplainer not available in this PyG version — "
              "using compliance reasoning only.")
        return []


# ═════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════

def _plot_explanation_cards(explanations: list, fraud_scores: list,
                             tx_ids: list):
    """
    Generate a visual card for each explained transaction showing
    fraud score gauge + reason bullets.
    Saves to outputs/explainability/explanation_cards.png
    """
    n     = len(explanations)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows))
    fig.suptitle("GNN Fraud Explanation Cards", fontsize=14,
                 fontweight="bold", y=1.02)

    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    flat_axes = [ax for row in axes for ax in row]

    for i, (exp, ax) in enumerate(zip(explanations, flat_axes)):
        score   = exp["fraud_score"]
        verdict = exp["verdict"]
        reasons = exp["reasons"]
        tx_id   = tx_ids[i]

        # Background colour by risk level
        if score >= 0.70:
            bg, fg = "#fdecea", "#c0392b"
        elif score >= 0.40:
            bg, fg = "#fef9e7", "#d35400"
        else:
            bg, fg = "#eafaf1", "#1e8449"

        ax.set_facecolor(bg)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Header
        ax.text(5, 9.3, f"Transaction #{tx_id}", ha="center",
                fontsize=10, fontweight="bold", color="#2c3e50")
        ax.text(5, 8.6, f"Fraud Score: {score:.3f}", ha="center",
                fontsize=12, fontweight="bold", color=fg)

        # Score bar
        ax.barh(7.8, score * 10, height=0.5, color=fg, alpha=0.7,
                left=0)
        ax.barh(7.8, 10,         height=0.5, color="#ecf0f1",
                left=0, zorder=0)
        ax.axvline(4.0, ymin=0.74, ymax=0.83, color="orange",
                   linewidth=1.5, linestyle="--")
        ax.axvline(7.0, ymin=0.74, ymax=0.83, color="red",
                   linewidth=1.5, linestyle="--")

        # Verdict
        ax.text(5, 7.1, verdict, ha="center", fontsize=8,
                color=fg, style="italic")

        # Reasons
        ax.text(0.3, 6.4, "Why flagged:", fontsize=8,
                fontweight="bold", color="#2c3e50")
        y_pos = 5.8
        for reason in reasons[:4]:   # max 4 reasons per card
            wrapped = reason[:70] + "…" if len(reason) > 70 else reason
            ax.text(0.3, y_pos, f"• {wrapped}", fontsize=7,
                    color="#2c3e50", va="top",
                    wrap=True)
            y_pos -= 1.2

    # Hide unused axes
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = "outputs/explainability/explanation_cards.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Explanation cards saved → {path}")


def _plot_risk_factor_heatmap(explanations: list, tx_ids: list):
    """
    Heatmap of risk factor scores across explained transactions.
    Saves to outputs/explainability/risk_factor_heatmap.png
    """
    factor_keys = [
        "customer_fraud_rate", "device_fraud_count",
        "email_fraud_rate",    "address_fraud_rate",
        "customer_tx_count",
    ]
    factor_labels = [
        "Customer\nFraud Rate", "Device\nFraud Count",
        "Email\nFraud Rate",    "Address\nFraud Rate",
        "Customer\nTx Count",
    ]

    matrix = []
    for exp in explanations:
        rf  = exp.get("risk_factors", {})
        row = []
        for k in factor_keys:
            val = rf.get(k, 0.0)
            row.append(float(val) if val is not None else 0.0)
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalise each column to [0,1] for visual comparison
    col_max = matrix.max(axis=0)
    col_max[col_max == 0] = 1
    norm_matrix = matrix / col_max

    fig, ax = plt.subplots(figsize=(10, max(3, len(tx_ids) * 0.6 + 2)))
    im = ax.imshow(norm_matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(factor_labels)))
    ax.set_xticklabels(factor_labels, fontsize=9)
    ax.set_yticks(range(len(tx_ids)))
    ax.set_yticklabels([f"TX#{t}" for t in tx_ids], fontsize=8)
    ax.set_title("Risk Factor Heatmap — Flagged Transactions",
                 fontsize=12, fontweight="bold")

    # Annotate cells with raw values
    for i in range(len(tx_ids)):
        for j in range(len(factor_keys)):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center",
                    va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Normalised Risk (0=low, 1=high)")
    plt.tight_layout()
    path = "outputs/explainability/risk_factor_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Risk factor heatmap saved → {path}")


# ═════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════

def run_explainability(model, graph, df: pd.DataFrame,
                        y_proba: np.ndarray, n_explain: int = 10):
    """
    Run full explainability pipeline.

    Args:
        model:      trained HeteroFraudGNN (in eval mode)
        graph:      HeteroData with node features
        df:         preprocessed dataframe (same row order as graph)
        y_proba:    fraud probabilities from train_gnn [N]
        n_explain:  number of high-confidence fraud cases to explain
    """
    print("\n" + "=" * 55)
    print("  STEP 11: Explainability Layer")
    print("=" * 55)

    os.makedirs("outputs/explainability", exist_ok=True)
    device = next(model.parameters()).device

    # ── Select transactions to explain ───────────────────────────────
    # Pick the top-N highest fraud scores for rich explanations
    top_fraud_idx = np.argsort(y_proba)[::-1][:n_explain]
    print(f"\n  Explaining top {n_explain} highest-confidence fraud predictions...")

    # ── Option A: GNNExplainer ────────────────────────────────────────
    print("\n  [Option A] GNNExplainer...")
    gnn_exp_results = run_gnn_explainer(
        model=model,
        graph=graph,
        target_tx_indices=top_fraud_idx.tolist(),
        device=device
    )

    # ── Option B: Compliance Reasoning ────────────────────────────────
    print("\n  [Option B] Compliance Reasoning Engine...")
    reasoning_engine = ComplianceReasoningEngine(df)

    explanations = []
    tx_ids       = []
    report_lines = [
        "=" * 55,
        "  FRAUD EXPLANATION REPORT",
        f"  Generated for top {n_explain} flagged transactions",
        "=" * 55,
        "",
    ]

    df_reset = df.reset_index(drop=True)

    # Check if TransactionID column exists and contains real IDs (not scaled floats)
    has_real_tx_id = (
        "TransactionID" in df_reset.columns and
        df_reset["TransactionID"].dtype in ["int64", "int32", "object"] and
        df_reset["TransactionID"].iloc[0] > 1000   # real IDs start at ~2987000
    )

    for tx_idx in top_fraud_idx:
        if tx_idx >= len(df_reset):
            continue

        row   = df_reset.iloc[tx_idx]
        # Use real TransactionID if available, otherwise just use the index
        if has_real_tx_id:
            tx_id = int(row["TransactionID"])
        else:
            tx_id = int(tx_idx)
        score      = float(y_proba[tx_idx])
        explanation = reasoning_engine.explain(row, score)

        explanations.append(explanation)
        tx_ids.append(tx_id)

        # Print to console
        report_block = reasoning_engine.format_report(tx_id, explanation)
        print(report_block)
        report_lines.append(report_block)
        report_lines.append("")

    # ── Save text report ──────────────────────────────────────────────
    report_path = "outputs/explainability/fraud_explanation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Explanation report saved → {report_path}")

    # ── Visualisations ────────────────────────────────────────────────
    print("\n  Generating visualisations...")
    _plot_explanation_cards(explanations, y_proba[top_fraud_idx], tx_ids)
    _plot_risk_factor_heatmap(explanations, tx_ids)

    print("\n  Explainability complete.")
    print(f"  Outputs → outputs/explainability/")

    return explanations
