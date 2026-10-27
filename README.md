# Graph-Based Fraud Detection

This repository implements reproducible PaySim fraud-detection experiments using four classifiers: Logistic Regression, Random Forest, XGBoost, and an MLP. Its graph workflow builds a deterministic, resource-bounded local sample; no graph database service is required.

## Setup

Use Python 3.11 or newer. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the PaySim CSV at the path configured by `data.raw_path` in `config/config.yaml` (the default is `data/raw/PS_20174392719_1491204439457_log.csv`). Do not commit the dataset or generated caches.

## Run

```powershell
python scripts/run_preprocessing.py
python scripts/run_temporal_features.py
python scripts/run_sampled_graph_features.py
python scripts/run_experiments.py
```

Or run the available stages in sequence:

```powershell
python run_pipeline.py
```

`run_pipeline.py` runs those stages in sequence. `run_experiments.py` evaluates the same sampled training rows for three feature sets: sampled tabular-only, tabular plus sampled graph, and tabular plus sampled graph and 24-hour temporal features. The edge cap and seed are configured in `config/config.yaml`. `run_temporal_features.py` also generates configured 6h and 72h sensitivity caches; the primary matrix uses the configured 24h window. `run_baselines.py` remains available for full-training-split tabular baselines.

The local graph implementation uses NetworkX weighted PageRank and weakly connected components (not Louvain). It samples only training-period edges; all models are fitted on those same sampled training rows, while validation and test remain the full chronological partitions. Treat sample-based graph results as resource-bounded experiments rather than full-graph estimates.

The notebooks in `notebooks/` are presentation/exploration entry points; reusable logic stays in `src/`. Open them with the repository root as the working directory, or run them from the notebook directory (their bootstrap handles either location).

## Methodology and artifacts

<!-- BEGIN MEASURED RESULTS -->
## Measured experiment results

**Status:** All 12 local custom-graph model/feature experiments completed.

- Dataset: 6,362,620 transactions; 8,213 fraud (0.1291%).
- Chronological boundaries by unique step: train through 520, validation through 631, test through 743. Rows: 6,082,007 train / 191,147 validation / 89,466 test.
- Reported PR-AUC, ROC-AUC, precision, recall, and F1 are test-set measurements. Threshold-dependent metrics use thresholds chosen on validation data only.
- Custom graph comparisons use the same deterministic sampled training rows for all models and feature sets. Validation and test rows remain unchanged.

| Model | Features | PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic | tabular_sampled | 0.807399 | 0.830875 | 0.674920 | 0.744822 | 0.985131 |
| Logistic | sampled_graph | 0.765795 | 0.833517 | 0.603834 | 0.700324 | 0.979015 |
| Logistic | sampled_temporal | 0.696202 | 0.766702 | 0.577476 | 0.658770 | 0.975141 |
| Random Forest | tabular_sampled | 0.997418 | 0.997590 | 0.992013 | 0.994794 | 0.999934 |
| Random Forest | sampled_graph | 0.996614 | 0.993605 | 0.992812 | 0.993208 | 0.999298 |
| Random Forest | sampled_temporal | 0.998459 | 0.996000 | 0.994409 | 0.995204 | 0.999964 |
| XGBoost | tabular_sampled | 0.968646 | 0.905721 | 0.897764 | 0.901725 | 0.999224 |
| XGBoost | sampled_graph | 0.970304 | 0.899696 | 0.945687 | 0.922118 | 0.998864 |
| XGBoost | sampled_temporal | 0.971133 | 0.913978 | 0.882588 | 0.898009 | 0.999262 |
| MLP | tabular_sampled | 0.740390 | 0.856828 | 0.621406 | 0.720370 | 0.974021 |
| MLP | sampled_graph | 0.751701 | 0.900878 | 0.573482 | 0.700830 | 0.963074 |
| MLP | sampled_temporal | 0.796303 | 0.904437 | 0.634984 | 0.746129 | 0.962149 |

### Supplemental temporal-only comparison

These runs use tabular plus 24-hour temporal features, without static graph features. They are reported separately and do not count toward the planned 12-run matrix.

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.659818 | 0.827586 | 0.498403 | 0.622134 | 0.971500 |
| Random Forest | 1.000000 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| XGBoost | 0.996539 | 0.971273 | 0.999201 | 0.985039 | 0.999954 |
| MLP | 0.877710 | 0.831137 | 0.805911 | 0.818329 | 0.994307 |

The local custom graph is built with NetworkX from a deterministic random sample of 50,000 training transactions, including 51 fraud rows and 98,145 accounts (seed 42).

All models are fit on these sampled training rows; the three feature sets are tabular-only, tabular plus sampled graph, and tabular plus sampled graph and 24-hour temporal features. Validation and test rows remain unchanged. The sampled graph uses weighted PageRank and weakly connected components, not full-graph Louvain communities.

### Supplemental full-training-data tabular baselines

These previously measured tabular runs used the full training split and are reported separately; compare sampled-graph matrix cells against `tabular_sampled` for a matched training sample.

| Model | PR-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic | 0.755176 | 0.815789 | 0.594249 | 0.687616 | 0.979714 |
| Random Forest | 1.000000 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| XGBoost | 0.996867 | 0.971984 | 0.997604 | 0.984628 | 0.999958 |
| MLP | 0.911705 | 0.732240 | 0.856230 | 0.789396 | 0.997222 |

The PaySim CSV was obtained from [Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. The raw dataset is not included in this repository.
<!-- END MEASURED RESULTS -->

- The train/validation/test split is based on ordered unique `step` values and stored in `data/processed/split.json`. All experiments use the same untouched chronological validation and test partitions.
- Tabular predictors exclude account IDs, `isFraud`, and `isFlaggedFraud`. Scaling is fit only on training rows.
- Temporal features use the half-open interval `[step - window, step)`. All transactions at a step receive features before any transaction from that step updates history.
- The production temporal feature command uses DuckDB window queries with bounded memory and disk spill, rather than retaining Python account histories for the full dataset. Its numeric `RANGE BETWEEN window PRECEDING AND 1 PRECEDING` frames represent `[step-window, step)`, excluding all transactions in the current hourly step.
- Graph edges are sampled reproducibly from rows at or before `train_end_step`. During training, degree, weighted-degree, counterparty, and unseen-account values are computed from a causal prefix of sampled edges; no same-step or future sampled edges enter those features. Validation and test graph features are computed from the frozen sampled training graph. Training centrality/community features use defaults because global graph centrality would expose later training edges to earlier transactions.
- Positive class weights are calculated from the training split only. Validation PR-AUC early-stops the MLP. Thresholds are selected on validation F1 and then frozen for test evaluation.
- Feature caches are Parquet under `data/features/`; splits and EDA summaries are under `data/processed/` and `results/metrics/`. Main test metrics are written to `results/metrics/main_results.csv`. Figures and per-model predictions are written under `results/`.
- The full experiment run also writes `results/metrics/research_summary.md`, which answers the five research questions using measured test results rather than assumed outcomes. ROC-AUC and F1 are undefined when validation or test lacks one class; in that case the experiment stops with a clear error instead of publishing misleading metrics.
- Generated artifacts and raw data are ignored by Git.

Run tests with:

```powershell
python -m pytest
```
