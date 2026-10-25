# Graph-Based Fraud Detection

This repository implements a reproducible PaySim fraud-detection experiment. It compares Logistic Regression, Random Forest, XGBoost, and an MLP across tabular, static graph, and temporal graph feature sets. The experiment does not assume graph features improve the result.

## Setup

Use Python 3.11 or newer. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the PaySim CSV at the path configured by `data.raw_path` in `config/config.yaml` (the default is `data/raw/PS_20174392719_1491204439457_log.csv`). Do not commit the dataset or generated caches.

Static graph features use Neo4j with the Graph Data Science plugin. Set the password named by `graph.password_env` (default `NEO4J_PASSWORD`) in the environment and check `graph.uri`, `graph.database`, and `graph.username`. The importer uses project-scoped `PaySimFraudAccount` nodes (also labelled `Account`) and `TRANSACTION` relationships; it only removes its own project-marked graph data. Use a database where the Neo4j GDS plugin is enabled.

## Run

```powershell
python scripts/run_preprocessing.py
python scripts/run_temporal_features.py
python scripts/run_graph_features.py
python scripts/run_experiments.py
```

Or run the available stages in sequence:

```powershell
python run_pipeline.py
```

`run_baselines.py` runs the tabular-only model comparison. `run_temporal_features.py` also generates configured 6h and 72h sensitivity caches; the primary matrix uses the configured 24h window.

The notebooks in `notebooks/` are presentation/exploration entry points; reusable logic stays in `src/`. Open them with the repository root as the working directory, or run them from the notebook directory (their bootstrap handles either location).

## Methodology and artifacts

- The train/validation/test split is based on ordered unique `step` values and stored in `data/processed/split.json`. All feature sets share those exact row partitions.
- Tabular predictors exclude account IDs, `isFraud`, and `isFlaggedFraud`. Scaling is fit only on training rows.
- Temporal features use the half-open interval `[step - window, step)`. All transactions at a step receive features before any transaction from that step updates history.
- Neo4j graph construction is capped at `train_end_step`. Validation and test static features therefore use no validation/test edges. For training rows, degree, weighted-degree, counterparty, and unseen-account features are computed from a causal transaction prefix. PageRank and community are set to their explicit defaults (`0` and `-1`) on training rows because a full training-graph centrality computation would expose later training edges to earlier transactions. Validation/test rows receive centrality and community values from the frozen training graph. This keeps the causal guarantee but means those two features have a training/serving distribution limitation that must be considered when interpreting results.
- Positive class weights are calculated from the training split only. Validation PR-AUC early-stops the MLP. Thresholds are selected on validation F1 and then frozen for test evaluation.
- Feature caches are Parquet under `data/features/`; splits and EDA summaries are under `data/processed/` and `results/metrics/`. Main test metrics are written to `results/metrics/main_results.csv`. Figures and per-model predictions are written under `results/`.
- The full experiment run also writes `results/metrics/research_summary.md`, which answers the five research questions using measured test results rather than assumed outcomes. ROC-AUC and F1 are undefined when validation or test lacks one class; in that case the experiment stops with a clear error instead of publishing misleading metrics.
- Generated artifacts and raw data are ignored by Git.

Run tests with:

```powershell
python -m pytest
```
