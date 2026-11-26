# Evaluations

Measured outputs of `python code/run_pipeline.py`. All scores are on the untouched chronological test period (steps 632–743: 89,466 transactions, 1,252 frauds).

Precision, recall, and F1 are reported for the fraud class only (`isFraud = 1`); thresholds are chosen on validation fraud F1 and frozen for test. PR-AUC is the threshold-free ranking metric.

| Path | Contents |
|---|---|
| `metrics/main_results.csv` | One row per model and feature set (28 runs): PR-AUC, ROC-AUC, fraud precision, recall, F1, threshold, fraud support, TP/FP/FN, training rows |
| `metrics/fraud_class_gains.csv` | Per-model change in fraud-class metrics when graph or temporal features are added |
| `metrics/ablation_gains.csv` | Sampled-track PR-AUC gains |
| `metrics/research_summary.md` | Research-question answers generated from the measured results |
| `metrics/*_importance_*.csv` | Random Forest and XGBoost feature importances |
| `metrics/eda_summary.json` | Dataset statistics |
| `figures/` | PR curves, ablation charts, feature importances, and EDA plots |
| `predictions/` | Per-row test predictions used by the demo (generated, not committed) |

Feature sets: `tabular_sampled`, `sampled_graph`, `sampled_temporal` (sampled track, 50,000 training rows) and `tabular`, `full_graph`, `temporal_only`, `full_temporal` (full-graph track, graph from all 6,082,007 training edges, models trained on all training frauds plus 10% of legitimate rows).
