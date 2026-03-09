"""
src/simulation/realtime_pipeline.py

Step 8: Real-Time Pipeline Simulation
--------------------------------------
Simulates a live transaction stream arriving one event at a time.

Architecture:
  1. Transaction event arrives  → EventQueue
  2. Graph updated              → Graphupdater
  3. Node memories updated      → NodeMemoryStore
  4. Embeddings computed        → trained HeteroFraudGNN
  5. Fraud score generated      → scorer
  6. Decision triggered         → DecisionEngine (flag / review / pass)

Run standalone:
    python -m src.simulation.realtime_pipeline
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import queue
import threading
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque, defaultdict
from datetime import datetime

from src.models.hetero_gnn import HeteroFraudGNN


# ═══════════════════════════════════════════════════════════════════════
# 1.  EVENT QUEUE
# ═══════════════════════════════════════════════════════════════════════

class TransactionEventQueue:
    """
    Thread-safe FIFO queue that receives raw transaction events.
    In production this would be a Kafka consumer or websocket stream.
    Here we simulate by pushing rows from the dataset one at a time.
    """

    def __init__(self, maxsize: int = 1000):
        self._q = queue.Queue(maxsize=maxsize)

    def push(self, event: dict):
        self._q.put(event, block=False)

    def pop(self, timeout: float = 1.0):
        return self._q.get(timeout=timeout)

    def size(self):
        return self._q.qsize()


# ═══════════════════════════════════════════════════════════════════════
# 2.  NODE MEMORY STORE
# ═══════════════════════════════════════════════════════════════════════

class NodeMemoryStore:
    """
    Lightweight in-memory store that tracks per-entity state.
    Mimics the memory module from Temporal Graph Networks (TGN).

    Tracks per customer / device / email / address:
      - transaction count
      - fraud count seen so far
      - rolling average transaction amount (last 10)
      - last seen timestamp
    """

    def __init__(self):
        self.memory = defaultdict(lambda: {
            "tx_count":     0,
            "fraud_count":  0,
            "amt_window":   deque(maxlen=10),
            "last_seen":    None,
        })

    def update(self, entity_key: str, amount: float,
               is_fraud: int, timestamp):
        m = self.memory[entity_key]
        m["tx_count"]   += 1
        m["fraud_count"] += int(is_fraud)
        m["amt_window"].append(amount)
        m["last_seen"]  = timestamp

    def get_risk_context(self, entity_key: str) -> dict:
        m = self.memory[entity_key]
        if m["tx_count"] == 0:
            return {"tx_count": 0, "fraud_rate": 0.0, "avg_amt": 0.0}
        return {
            "tx_count":  m["tx_count"],
            "fraud_rate": m["fraud_count"] / m["tx_count"],
            "avg_amt":   float(np.mean(m["amt_window"])) if m["amt_window"] else 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════
# 3.  DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Converts raw fraud probability + memory context into an action.

    Thresholds (tunable):
      score >= FLAG_THRESHOLD   → BLOCK   (high confidence fraud)
      score >= REVIEW_THRESHOLD → REVIEW  (send to analyst queue)
      score <  REVIEW_THRESHOLD → PASS    (allow transaction)

    Memory boosting: if the customer already has a high historical
    fraud rate, the effective score is boosted before thresholding.
    """

    FLAG_THRESHOLD   = 0.70
    REVIEW_THRESHOLD = 0.40

    def decide(self, fraud_prob: float, memory_context: dict) -> dict:
        # Boost score if entity has history of fraud
        historical_fraud_rate = memory_context.get("fraud_rate", 0.0)
        boosted_score = min(1.0, fraud_prob + 0.15 * historical_fraud_rate)

        if boosted_score >= self.FLAG_THRESHOLD:
            action = "BLOCK"
        elif boosted_score >= self.REVIEW_THRESHOLD:
            action = "REVIEW"
        else:
            action = "PASS"

        return {
            "raw_score":     round(fraud_prob, 4),
            "boosted_score": round(boosted_score, 4),
            "action":        action,
        }


# ═══════════════════════════════════════════════════════════════════════
# 4.  REAL-TIME PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class RealTimeFraudPipeline:
    """
    Orchestrates all components into a streaming simulation.
    """

    def __init__(self, model: HeteroFraudGNN, graph,
                 feature_cols: list, device: torch.device):

        self.model        = model
        self.graph        = graph
        self.feature_cols = feature_cols
        self.device       = device
        self.event_queue  = TransactionEventQueue()
        self.memory_store = NodeMemoryStore()
        self.decision_eng = DecisionEngine()
        self.results      = []

        self.model.eval()

    # ------------------------------------------------------------------
    # Score a single transaction tensor against the trained model.
    # We pass it through the GNN via the full graph — in production
    # you'd use a subgraph neighbourhood; here full-graph is fine.
    # ------------------------------------------------------------------
    def _score_transaction(self, tx_index: int) -> float:
        with torch.no_grad():
            probs = self.model(self.graph)   # [N] fraud probabilities
        return float(probs[tx_index].cpu())

    # ------------------------------------------------------------------
    # Process one event end-to-end
    # ------------------------------------------------------------------
    def _process_event(self, event: dict) -> dict:
        tx_idx     = event["tx_index"]
        customer   = str(event.get("card1",         "unknown"))
        device     = str(event.get("DeviceInfo",    "unknown"))
        email      = str(event.get("P_emaildomain", "unknown"))
        address    = str(event.get("addr1",         "unknown"))
        amount     = float(event.get("TransactionAmt", 0.0))
        label      = int(event.get("isFraud", -1))
        timestamp  = event.get("TransactionDT", 0)

        # Step 4: compute embedding + fraud score
        fraud_prob = self._score_transaction(tx_idx)

        # Step 3: fetch memory context BEFORE updating
        ctx = self.memory_store.get_risk_context(f"customer_{customer}")

        # Step 6: decision
        decision = self.decision_eng.decide(fraud_prob, ctx)

        # Step 3: update memory AFTER decision (simulate online update)
        predicted_fraud = 1 if decision["action"] == "BLOCK" else 0
        for key in [f"customer_{customer}", f"device_{device}",
                    f"email_{email}", f"address_{address}"]:
            self.memory_store.update(key, amount, predicted_fraud, timestamp)

        return {
            "tx_index":     tx_idx,
            "amount":       amount,
            "fraud_prob":   fraud_prob,
            "boosted_score": decision["boosted_score"],
            "action":       decision["action"],
            "true_label":   label,
            "timestamp":    timestamp,
        }

    # ------------------------------------------------------------------
    # Producer: push N transactions into the event queue
    # ------------------------------------------------------------------
    def _producer(self, events: list, speed: float):
        for ev in events:
            self.event_queue.push(ev)
            time.sleep(speed)   # simulate inter-arrival time

    # ------------------------------------------------------------------
    # Consumer: pull events and process them
    # ------------------------------------------------------------------
    def _consumer(self, n_events: int):
        processed = 0
        while processed < n_events:
            try:
                event = self.event_queue.pop(timeout=2.0)
                result = self._process_event(event)
                self.results.append(result)
                processed += 1

                action_color = {"BLOCK": "🔴", "REVIEW": "🟡", "PASS": "🟢"}
                icon = action_color.get(result["action"], "⚪")
                print(
                    f"  {icon} TX#{result['tx_index']:>7} | "
                    f"${result['amount']:>8.2f} | "
                    f"Score: {result['fraud_prob']:.3f} | "
                    f"{result['action']:<6} | "
                    f"True: {'FRAUD' if result['true_label'] == 1 else 'legit'}"
                )
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Main entry: run simulation
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, n_events: int = 50,
            simulation_speed: float = 0.05):
        """
        Args:
            df:               preprocessed dataframe
            n_events:         number of transactions to stream
            simulation_speed: seconds between arrivals (lower = faster)
        """
        print(f"\n{'='*60}")
        print(f"  REAL-TIME FRAUD PIPELINE  —  streaming {n_events} transactions")
        print(f"{'='*60}\n")

        # Sample a mix of fraud and non-fraud for an interesting demo
        fraud_sample    = df[df["isFraud"] == 1].head(n_events // 4)
        non_fraud_sample = df[df["isFraud"] == 0].head(n_events - len(fraud_sample))
        sample_df = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42)

        # Build event list
        events = []
        for local_pos, (orig_idx, row) in enumerate(sample_df.iterrows()):
            ev = row.to_dict()
            ev["tx_index"] = local_pos   # position in this sample
            events.append(ev)

        # Run producer in background thread, consumer in main thread
        producer_thread = threading.Thread(
            target=self._producer, args=(events, simulation_speed), daemon=True
        )
        producer_thread.start()
        self._consumer(n_events=len(events))
        producer_thread.join()

        self._print_summary()
        self._save_visualisation()

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def _print_summary(self):
        df = pd.DataFrame(self.results)
        total   = len(df)
        blocked = (df["action"] == "BLOCK").sum()
        review  = (df["action"] == "REVIEW").sum()
        passed  = (df["action"] == "PASS").sum()
        true_fraud = (df["true_label"] == 1).sum()

        correct_blocks = ((df["action"] == "BLOCK") & (df["true_label"] == 1)).sum()
        missed_fraud   = ((df["action"] == "PASS")  & (df["true_label"] == 1)).sum()

        print(f"\n{'='*60}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Total processed  : {total}")
        print(f"  🔴 BLOCKED        : {blocked}  ({100*blocked/total:.1f}%)")
        print(f"  🟡 REVIEW         : {review}   ({100*review/total:.1f}%)")
        print(f"  🟢 PASSED         : {passed}  ({100*passed/total:.1f}%)")
        print(f"\n  True fraud in sample : {true_fraud}")
        print(f"  Correctly blocked    : {correct_blocks}")
        print(f"  Missed (passed)      : {missed_fraud}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def _save_visualisation(self):
        os.makedirs("outputs/simulation", exist_ok=True)
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Real-Time Fraud Pipeline — Simulation Results", fontsize=14)

        # -- Plot 1: Decision distribution
        action_counts = df["action"].value_counts().reindex(
            ["BLOCK", "REVIEW", "PASS"], fill_value=0
        )
        colors = ["#e74c3c", "#f39c12", "#2ecc71"]
        axes[0].bar(action_counts.index, action_counts.values, color=colors)
        axes[0].set_title("Decision Distribution")
        axes[0].set_ylabel("Count")
        for i, v in enumerate(action_counts.values):
            axes[0].text(i, v + 0.3, str(v), ha='center', fontweight='bold')

        # -- Plot 2: Fraud score distribution by true label
        fraud_scores  = df[df["true_label"] == 1]["fraud_prob"]
        legit_scores  = df[df["true_label"] == 0]["fraud_prob"]
        axes[1].hist(legit_scores,  bins=20, alpha=0.6, color="#3498db", label="Legit")
        axes[1].hist(fraud_scores,  bins=20, alpha=0.6, color="#e74c3c", label="Fraud")
        axes[1].axvline(DecisionEngine.REVIEW_THRESHOLD, color='orange',
                        linestyle='--', label=f"Review ({DecisionEngine.REVIEW_THRESHOLD})")
        axes[1].axvline(DecisionEngine.FLAG_THRESHOLD,   color='red',
                        linestyle='--', label=f"Block ({DecisionEngine.FLAG_THRESHOLD})")
        axes[1].set_title("Fraud Score Distribution")
        axes[1].set_xlabel("Fraud Probability")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        # -- Plot 3: Rolling fraud rate over time
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        df_sorted["is_blocked"] = (df_sorted["action"] == "BLOCK").astype(int)
        rolling_rate = df_sorted["is_blocked"].rolling(window=10, min_periods=1).mean()
        axes[2].plot(rolling_rate, color="#9b59b6", linewidth=2)
        axes[2].set_title("Rolling Block Rate (window=10)")
        axes[2].set_xlabel("Transaction #")
        axes[2].set_ylabel("Block Rate")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = "outputs/simulation/realtime_pipeline_results.png"
        plt.savefig(path, dpi=150)
        print(f"  Simulation visualisation saved → {path}")


# ═══════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT  (called from main.py)
# ═══════════════════════════════════════════════════════════════════════

def run_realtime_simulation(graph, df: pd.DataFrame,
                             feature_cols: list, model_kwargs: dict):
    """
    Public function called from main.py

    Args:
        graph:         trained HeteroData graph with node features
        df:            original preprocessed dataframe
        feature_cols:  list of feature column names used in training
        model_kwargs:  dict with 'hidden_dim' and 'in_channels_dict'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph  = graph.to(device)

    # Re-instantiate and reload isn't needed — we receive the live graph.
    # Build a fresh model in eval mode using the same architecture.
    model = HeteroFraudGNN(
        metadata=graph.metadata(),
        hidden_dim=model_kwargs["hidden_dim"],
        in_channels_dict=model_kwargs["in_channels_dict"]
    ).to(device)

    # NOTE: In a full pipeline you'd load saved weights here:
    # model.load_state_dict(torch.load("outputs/models/gnn_weights.pt"))
    # For now we run with the current session's trained weights passed in.
    if "state_dict" in model_kwargs:
        model.load_state_dict(model_kwargs["state_dict"])

    pipeline = RealTimeFraudPipeline(
        model=model,
        graph=graph,
        feature_cols=feature_cols,
        device=device
    )

    results_df = pipeline.run(df, n_events=50, simulation_speed=0.01)
    return results_df