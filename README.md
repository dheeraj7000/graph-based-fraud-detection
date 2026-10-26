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

<!-- BEGIN MEASURED RESULTS -->
## Measured experiment results

**Status:** Partial run: 4/12 planned experiments completed. Results below are measured; missing matrix cells are not run. Do not interpret the table as a completed ablation.

- Static and temporal graph-feature experiments were not run because no Neo4j/GDS service was configured in the execution environment.
- Dataset: 6,362,620 transactions; 8,213 fraud (0.1291%).
- Chronological boundaries by unique step: train through 520, validation through 631, test through 743. Rows: 6,082,007 train / 191,147 validation / 89,466 test.
- Reported PR-AUC, ROC-AUC, precision, recall, and F1 are test-set measurements. Threshold-dependent metrics use thresholds chosen on validation data only.

| Model | Features | PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic | tabular | 0.755176 | 0.815789 | 0.594249 | 0.687616 | 0.979714 |
| Logistic | static | Not run | Not run | Not run | Not run | Not run |
| Logistic | temporal | Not run | Not run | Not run | Not run | Not run |
| Random Forest | tabular | 1.000000 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| Random Forest | static | Not run | Not run | Not run | Not run | Not run |
| Random Forest | temporal | Not run | Not run | Not run | Not run | Not run |
| XGBoost | tabular | 0.996867 | 0.971984 | 0.997604 | 0.984628 | 0.999958 |
| XGBoost | static | Not run | Not run | Not run | Not run | Not run |
| XGBoost | temporal | Not run | Not run | Not run | Not run | Not run |
| MLP | tabular | 0.911705 | 0.732240 | 0.856230 | 0.789396 | 0.997222 |
| MLP | static | Not run | Not run | Not run | Not run | Not run |
| MLP | temporal | Not run | Not run | Not run | Not run | Not run |

The PaySim CSV was obtained from [Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. The raw dataset is not included in this repository.
<!-- END MEASURED RESULTS -->

- The train/validation/test split is based on ordered unique `step` values and stored in `data/processed/split.json`. All feature sets share those exact row partitions.
- Tabular predictors exclude account IDs, `isFraud`, and `isFlaggedFraud`. Scaling is fit only on training rows.
- Temporal features use the half-open interval `[step - window, step)`. All transactions at a step receive features before any transaction from that step updates history.
- The production temporal feature command uses DuckDB window queries with bounded memory and disk spill, rather than retaining Python account histories for the full dataset. Its numeric `RANGE BETWEEN window PRECEDING AND 1 PRECEDING` frames represent `[step-window, step)`, excluding all transactions in the current hourly step.
- Neo4j graph construction is capped at `train_end_step`. Validation and test static features therefore use no validation/test edges. For training rows, degree, weighted-degree, counterparty, and unseen-account features are computed from a causal transaction prefix. PageRank and community are set to their explicit defaults (`0` and `-1`) on training rows because a full training-graph centrality computation would expose later training edges to earlier transactions. Validation/test rows receive centrality and community values from the frozen training graph. This keeps the causal guarantee but means those two features have a training/serving distribution limitation that must be considered when interpreting results.
- Positive class weights are calculated from the training split only. Validation PR-AUC early-stops the MLP. Thresholds are selected on validation F1 and then frozen for test evaluation.
- Feature caches are Parquet under `data/features/`; splits and EDA summaries are under `data/processed/` and `results/metrics/`. Main test metrics are written to `results/metrics/main_results.csv`. Figures and per-model predictions are written under `results/`.
- The full experiment run also writes `results/metrics/research_summary.md`, which answers the five research questions using measured test results rather than assumed outcomes. ROC-AUC and F1 are undefined when validation or test lacks one class; in that case the experiment stops with a clear error instead of publishing misleading metrics.
- Generated artifacts and raw data are ignored by Git.

Run tests with:

```powershell
python -m pytest
```
