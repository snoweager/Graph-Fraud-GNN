# main.py

import os
import json

from src.data_processing.load_data import load_data
from src.data_processing.preprocess import preprocess_data
from src.rule_engine.rules import RuleBasedEngine
from src.graph_builder.build_graph import build_hetero_graph
from src.graph_builder.add_features import attach_node_features
from src.evaluation.metrics import evaluate_model
from src.monitoring.graph_stats import save_graph_statistics
from src.training.temporal_split import temporal_train_test_split
from src.monitoring.temporal_plots import save_temporal_distribution
from src.training.train_gnn import train_gnn
from src.simulation.realtime_pipeline import run_realtime_simulation
from src.evaluation.metrics import evaluate_model, compare_models
from src.explainability.gnn_explainer import run_explainability
from src.monitoring.drift_detection import run_drift_detection
from src.evaluation.comparison_study import run_comparison_study

def main():

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/graphs", exist_ok=True)

    # ----------------------------------------------------------------
    print("=" * 50)
    print("STEP 1: Loading Data")
    print("=" * 50)

    df = load_data()

    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 2: Preprocessing Data")
    print("=" * 50)

    df_processed, scaler, encoders = preprocess_data(df)

    print("\nSTEP 2.5: Temporal Train-Test Split")

    train_df, test_df = temporal_train_test_split(df_processed)
    save_temporal_distribution(train_df, test_df)

    # Target labels — derived from df_processed, not raw df
    y_true = df_processed["isFraud"].values

    # Feature columns = everything except label and ID columns
    feature_cols = [
        c for c in df_processed.columns
        if c not in ("isFraud", "TransactionID", "TransactionDT")
    ]

    print("\nFraud distribution:")
    print(df_processed["isFraud"].value_counts())

    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 3: Rule-Based Engine Baseline")
    print("=" * 50)

    engine = RuleBasedEngine()
    engine.fit(df)
    y_pred_rules  = engine.predict(df)
    y_proba_rules = engine.predict_proba(df)

    results_rule = evaluate_model(
        y_true=y_true,
        y_pred=y_pred_rules,
        y_proba=y_proba_rules,
        model_name="Rule-Based Engine"
    )
    with open("outputs/metrics/rule_engine_results.json", "w") as f:
        json.dump(results_rule, f, indent=4)

    print("\nMetrics saved to outputs/metrics/")

    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 4: Building Graph")
    print("=" * 50)

    graph_data = build_hetero_graph(df_processed)
    print(graph_data)

    with open("outputs/graphs/graph_summary.txt", "w") as f:
        f.write(str(graph_data))

    print("\nGraph summary saved.")

    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 5: Attaching Node Features")
    print("=" * 50)

    graph_data = attach_node_features(graph_data, df_processed)

    # ----------------------------------------------------------------
    # STEP 6 label is printed inside train_gnn
    y_pred_gnn, y_proba_gnn, model_kwargs = train_gnn(graph_data)

    results_gnn = evaluate_model(
        y_true=y_true,
        y_pred=y_pred_gnn,
        y_proba=y_proba_gnn,
        model_name="Heterogeneous GNN"
    )
    # ----------------------------------------------------------------
    print("\nSTEP 6: Saving Graph Statistics")

    save_graph_statistics(graph_data)
    print(graph_data)

    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("STEP 8: Real-Time Pipeline Simulation")
    print("=" * 50)

    run_realtime_simulation(
        graph=graph_data,
        df=df_processed,        # FIX: use df_processed, not raw df
        feature_cols=feature_cols,
        model_kwargs=model_kwargs
    )

    # ── Step 10: Side-by-side comparison ─────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 10: Evaluation Framework — Model Comparison")
    print("=" * 50)

    compare_models(
        y_true=y_true,
        results_rule=results_rule,
        results_gnn=results_gnn,
        y_proba_rule=y_proba_rules,
        y_proba_gnn=y_proba_gnn,
    )

    print("\n" + "="*50)
    print("STEP 11: Explainability Layer")
    print("="*50)

    run_explainability(
        model=model_kwargs["model"],   # see note below
        graph=graph_data,
        df=df_processed,
        y_proba=y_proba_gnn,
        n_explain=10
    )

    print("\n" + "="*50)
    print("STEP 12: Drift Detection")
    print("="*50)

    run_drift_detection(
        df=df_processed,
        y_proba=y_proba_gnn,
        n_windows=6
    )

    print("\n" + "="*50)
    print("STEP 13: Final Comparison Study")
    print("="*50)

    run_comparison_study(
        df=df_processed,
        y_true=y_true,
        results_rule=results_rule,
        results_gnn=results_gnn,
        y_proba_rule=y_proba_rules,
        y_proba_gnn=y_proba_gnn,
    )
    # ----------------------------------------------------------------
    
    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()