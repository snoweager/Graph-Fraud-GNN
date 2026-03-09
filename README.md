# 🧠 Real-Time Heterogeneous Temporal GNN — Fraud Detection System

> **Proving the thesis:** A Temporal Heterogeneous GNN outperforms rule-based fraud engines in adaptability, contextual awareness, and false-positive reduction.

---

## 📌 Project Overview

This is a **production-grade fraud intelligence system** built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (590,540 transactions). It goes far beyond a standard Kaggle notebook — combining graph neural networks, explainability, drift detection, a REST API, and a visual dashboard into a complete end-to-end pipeline.

| Component | Description |
|---|---|
| 🕸️ **Heterogeneous GNN** | Multi-relational graph with 5 node types, trained with Focal Loss |
| 🔵 **Rule-Based Baseline** | Vectorized 6-rule engine mimicking legacy fraud systems |
| 🔍 **Explainability** | Compliance-grade reasoning + GNNExplainer subgraph analysis |
| 📡 **Drift Detection** | PSI + KL Divergence across temporal windows |
| ⚡ **Real-Time API** | FastAPI scoring endpoint with BLOCK / REVIEW / PASS decisions |
| 📊 **Visual Dashboard** | Self-contained HTML report tying all outputs together |

---

## 🏆 Key Results

| Metric | Rule Engine | GNN | Winner |
|---|---|---|---|
| ROC-AUC | 0.3894 | 0.6792 | 🟢 GNN |
| F1 Score | 0.0485 | 0.1484 | 🟢 GNN |
| False Positive Rate | High | Lower | 🟢 GNN |
| Total Cost (FP+FN) | $5,097,060 | $3,124,930 | 🟢 GNN |
| **Cost Saving** | — | **$1,972,130** | 🟢 GNN |
| FP Reduction | — | **358,883 transactions** | 🟢 GNN |

> Cost model: $10 per false positive (wrongly blocked legit transaction) + $150 per false negative (missed fraud)

---

## 📁 Project Structure

```
graph-fraud-gnn/
│
├── data/
│   ├── raw/                          # Original IEEE-CIS CSV files
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   ├── test_transaction.csv
│   │   └── test_identity.csv
│   └── processed/                    # Cleaned/preprocessed data
│
├── src/
│   ├── data_processing/
│   │   ├── load_data.py              # Merge transaction + identity datasets
│   │   └── preprocess.py            # Feature selection, scaling, encoding
│   │
│   ├── graph_builder/
│   │   ├── build_graph.py           # Heterogeneous graph construction
│   │   └── add_features.py          # Attach node feature matrices
│   │
│   ├── models/
│   │   └── hetero_gnn.py            # HeteroFraudGNN architecture
│   │
│   ├── training/
│   │   ├── train_gnn.py             # Focal Loss training loop + threshold tuning
│   │   └── temporal_split.py        # Chronological train/test split
│   │
│   ├── rule_engine/
│   │   └── rules.py                 # Vectorized 6-rule fraud engine
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # evaluate_model() + compare_models()
│   │   └── comparison_study.py      # Full 6-part comparison study
│   │
│   ├── explainability/
│   │   └── gnn_explainer.py         # GNNExplainer + Compliance Reasoning Engine
│   │
│   ├── simulation/
│   │   └── realtime_pipeline.py     # Event queue + streaming fraud simulation
│   │
│   ├── monitoring/
│   │   ├── drift_detection.py       # PSI + KL Divergence drift monitoring
│   │   ├── graph_stats.py           # Graph statistics + visualisations
│   │   ├── temporal_plots.py        # Train/test split visualisation
│   │   └── dashboard.py             # HTML dashboard generator
│   │
│   └── api/
│       └── scoring_api.py           # FastAPI real-time scoring endpoint
│
├── outputs/                         # ← ALL RESULTS SAVED HERE (see below)
├── notebooks/
├── tests/
├── main.py                          # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## 📊 Where to See Results

All outputs are saved automatically when you run `main.py`. Here's exactly where to find everything:

### 🔵 Rule-Based Engine
| Output | Path |
|---|---|
| Metrics (JSON) | `outputs/metrics/rule_engine_results.json` |
| Rule trigger rates | Printed to console during Step 3 |

### 🧠 GNN Training
| Output | Path |
|---|---|
| Training loss curve | `outputs/training/gnn_training_loss.png` |
| ROC-AUC + threshold | Printed to console during Step 6 |

### 🕸️ Graph Statistics
| Output | Path |
|---|---|
| Node type distribution | `outputs/graphs/node_distribution.png` |
| Edge type distribution | `outputs/graphs/edge_distribution.png` |
| Temporal train/test split | `outputs/graphs/temporal_split_distribution.png` |
| Graph structure summary | `outputs/graphs/graph_summary.txt` |

### ⚖️ Model Comparison
| Output | Path |
|---|---|
| Side-by-side metrics chart | `outputs/evaluation/model_comparison.png` |
| Comparison JSON | `outputs/metrics/model_comparison.json` |

### 🔍 Explainability
| Output | Path |
|---|---|
| Fraud explanation cards | `outputs/explainability/explanation_cards.png` |
| Risk factor heatmap | `outputs/explainability/risk_factor_heatmap.png` |
| Compliance text report | `outputs/explainability/fraud_explanation_report.txt` |

### 📡 Drift Detection
| Output | Path |
|---|---|
| Score distribution over time | `outputs/monitoring/drift_score_distribution.png` |
| PSI + KL divergence chart | `outputs/monitoring/psi_over_time.png` |
| Drift JSON report | `outputs/monitoring/drift_report.json` |

### 🧪 Comparison Study (Step 13)
| Output | Path |
|---|---|
| 4-panel comparison summary | `outputs/comparison/comparison_summary.png` |
| Adaptability + cold-start tests | `outputs/comparison/stress_tests.png` |
| Full study JSON | `outputs/comparison/comparison_study.json` |

### 🎮 Real-Time Simulation
| Output | Path |
|---|---|
| Pipeline results chart | `outputs/simulation/realtime_pipeline_results.png` |

### 📊 Visual Dashboard ← **Start here**
| Output | Path |
|---|---|
| **Full HTML dashboard** | `outputs/dashboard/fraud_intelligence_dashboard.html` |

> 💡 **Tip:** Open `outputs/dashboard/fraud_intelligence_dashboard.html` in any browser for a complete visual overview of the entire project — no server needed, fully self-contained.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/graph-fraud-gnn.git
cd graph-fraud-gnn
```

### 2. Set up environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) and place CSV files in `data/raw/`:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 4. Run the full pipeline
```bash
py main.py
```

The pipeline runs all 13 steps automatically (~10–20 min depending on hardware).

### 5. Start the scoring API (optional)
```bash
pip install fastapi uvicorn
uvicorn src.api.scoring_api:app --reload --port 8000
```
Then open `http://localhost:8000/docs` for the interactive Swagger UI.

**Example API call:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 850.0, "card1": 12345, "DeviceInfo": "unknown",
       "P_emaildomain": "gmail.com", "R_emaildomain": "hotmail.com", "addr1": 315}'
```

---

## 🔬 System Architecture

```
Raw Data (590,540 transactions)
        │
        ▼
┌─────────────────┐     ┌──────────────────────┐
│  Preprocessing  │     │  Rule-Based Baseline  │
│  (67 features)  │     │  (6 vectorized rules) │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         │
┌─────────────────────────┐        │
│   Heterogeneous Graph   │        │
│  ┌─────────┐            │        │
│  │customer │──makes──►  │        │
│  │ 13,553  │  txn        │        │
│  └─────────┘            │        │
│  ┌─────────┐            │        │
│  │ device  │─used_in─►  │        │
│  │  1,787  │  txn        │        │
│  └─────────┘            │        │
│  ┌─────────┐            │        │
│  │  email  │─linked_►   │        │
│  │   60    │  txn        │        │
│  └─────────┘            │        │
│  ┌─────────┐            │        │
│  │address  │─located►   │        │
│  │   332   │  txn        │        │
│  └─────────┘            │        │
│  590,540 transactions   │        │
└──────────┬──────────────┘        │
           │                       │
           ▼                       │
┌──────────────────────┐           │
│  HeteroFraudGNN      │           │
│  Input Projection    │           │
│  → SAGEConv Layer 1  │           │
│  → SAGEConv Layer 2  │           │
│  → MLP Classifier    │           │
│  → Focal Loss        │           │
└──────────┬───────────┘           │
           │                       │
           ▼                       ▼
┌──────────────────────────────────────┐
│         Evaluation Framework         │
│  Metrics · Cost Model · ROC Curves   │
└──────────┬───────────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌──────────────┐
│Explain- │  │    Drift     │
│ability  │  │  Detection   │
│GNNExp + │  │  PSI + KL    │
│Reasoning│  │  Divergence  │
└─────────┘  └──────────────┘
           │
           ▼
┌──────────────────────┐
│   FastAPI Scoring    │
│   /score endpoint    │
│   BLOCK/REVIEW/PASS  │
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│   HTML Dashboard     │
│  (self-contained)    │
└──────────────────────┘
```

---

## 🧩 Graph Schema

| Node Type | Count | Features |
|---|---|---|
| `transaction` | 590,540 | 66 (amount, time, device hash, etc.) |
| `customer` | 13,553 | 16 (aggregated behavioural features) |
| `device` | 1,787 | 16 |
| `email` | 60 | 16 |
| `address` | 332 | 16 |

| Edge Type | Count | Direction |
|---|---|---|
| `customer → makes → transaction` | 590,540 | Forward |
| `device → used_in → transaction` | 590,540 | Forward |
| `email → linked_to → transaction` | 590,540 | Forward |
| `address → located_at → transaction` | 590,540 | Forward |
| + 4 reverse edge types | 590,540 each | Backward |

---

## 🔬 Model Details

### HeteroFraudGNN
- **Architecture:** 2-layer Heterogeneous GraphSAGE with per-node-type BatchNorm
- **Input projection:** Linear(raw_dim → 64) per node type — handles variable feature sizes
- **Message passing:** `HeteroConv` with `SAGEConv` per edge type, `sum` aggregation
- **Reverse edges:** All 4 edge types include reverse for bidirectional updates
- **Classifier:** `Linear(64→64) → BN → ReLU → Dropout(0.4) → Linear(64→32) → BN → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid`
- **Loss:** Focal Loss (α=0.25, γ=2.0) — handles 96.5%/3.5% class imbalance
- **Threshold:** Tuned via precision-recall curve (not fixed at 0.5)

### Rule-Based Engine
6 vectorized rules with weighted scoring:
1. High transaction amount (top 5%) — weight 1.0
2. Rare device (seen < 50 times) — weight 1.0
3. Email domain mismatch (purchaser ≠ recipient) — weight 1.0
4. Burst transactions (>8 in rolling window) — weight 1.5
5. New device + high amount (compound) — weight 1.5
6. Credit card type — weight 1.0

---

## 📡 Drift Detection

Monitors fraud pattern evolution across 6 temporal windows:

| Status | PSI | KL | Action |
|---|---|---|---|
| ✅ Stable | < 0.10 | < 0.05 | Continue |
| ⚠️ Monitor | 0.10 – 0.25 | 0.05 – 0.20 | Increase monitoring |
| 🔴 Retrain | ≥ 0.25 | ≥ 0.20 | Trigger retraining |

---

## 🧪 Stress Tests

### Test 5 — Adaptability (New Fraud Pattern)
Simulates high-value transactions from previously clean devices — a pattern not seen during training. Tests whether the GNN generalises via graph neighbourhood context vs rule engine's rigid thresholds.

### Test 6 — Cold-Start Device Attack
Fraudsters using brand-new devices (seen only once). Rule engines are blind here since they rely on device frequency history. GNN can propagate risk signals from customer/email/address neighbours.

---

## 📦 Requirements

```
torch
torch-geometric
pandas
numpy
scikit-learn
matplotlib
scipy
fastapi
uvicorn
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🎯 What This Demonstrates

| Skill | Evidence |
|---|---|
| Graph Machine Learning | Heterogeneous GNN with multi-relational message passing |
| Production ML thinking | Focal Loss, threshold tuning, BatchNorm, gradient clipping |
| System design | Event queue, streaming simulation, REST API |
| Regulatory/compliance | Explainability layer with human-readable fraud reasons |
| Evaluation rigour | Cost model, FPR, FCR, stress tests, drift detection |
| Legacy system comparison | Rule engine baseline with fair side-by-side evaluation |

---

## 📄 License

MIT License — feel free to use, modify, and build on this.

---

*Built with PyTorch Geometric · FastAPI · Pandas · Scikit-learn*
