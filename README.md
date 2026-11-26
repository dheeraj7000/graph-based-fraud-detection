# Fraud Detection with Graph Databases

CSE 573, Fall 2026, Group 19: Vatsal Joshi, Dheeraj Kumar, Yaksh Pancholi, Lovinder Singh, Samarth Ogae.

This repository detects fraud in the PaySim mobile-money dataset (6.36M transactions, 0.129% fraud) using transaction features, 24-hour account activity, and features from the account graph, with four classifiers: Logistic Regression, Random Forest, XGBoost, and an MLP. The account graph is held in-process as SciPy sparse matrices, with DuckDB for windowed history queries, so no graph database server is required. An interactive Streamlit app presents the results.

## Repository layout

| Folder | Contents |
|---|---|
| [`code/`](code) | Pipeline (`run_pipeline.py`), reusable modules (`src/`), stage scripts (`scripts/`), demo app (`demo/`), tests (`tests/`), notebooks (`notebooks/`) |
| [`data/`](data) | Dataset description, download steps, and a committed 20,000-row sample; see [`data/README.md`](data/README.md) |
| [`evaluations/`](evaluations) | Measured metrics (`metrics/`), figures (`figures/`), and the research summary; see [`evaluations/README.md`](evaluations/README.md) |
| [`config/`](config) | Dataset path, sampling, and model settings |
| [`docs/`](docs) | Project guide and demo walkthrough |
| [`presentations/`](presentations) | Project presentation, demo presentation, and demo screenshots |

All commands run from the repository root.

## Setup

Use Python 3.11 or newer. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the PaySim CSV at the path configured by `data.raw_path` in `config/config.yaml` (the default is `data/raw/PS_20174392719_1491204439457_log.csv`); [`data/README.md`](data/README.md) explains how to download it. Do not commit the dataset or generated caches.

## Run

```powershell
python code/scripts/run_preprocessing.py
python code/scripts/run_temporal_features.py
python code/scripts/run_sampled_graph_features.py
python code/scripts/run_experiments.py
python code/scripts/run_full_graph_features.py
python code/scripts/run_full_graph_experiments.py
python code/scripts/prepare_demo.py
```

Or run every stage in sequence:

```powershell
python code/run_pipeline.py
```

Set `graph.full_graph: false` in `config/config.yaml` to skip the two full-graph stages. `run_experiments.py` evaluates the same 50,000 sampled training rows for three feature sets: tabular only, tabular plus sampled graph, and tabular plus sampled graph and 24-hour temporal features. `run_temporal_features.py` also generates 6h and 72h sensitivity caches; the main experiments use the 24h window.

`run_full_graph_features.py` builds the account graph from every training-period edge (6,082,007 edges, 8,669,206 accounts), not a sample. `run_full_graph_experiments.py` then fits all four models for four feature sets: `tabular`, `full_graph` (tabular plus full graph), `temporal_only` (tabular plus 24-hour temporal), and `full_temporal` (tabular plus full graph and 24-hour temporal). Comparing `full_graph` with `tabular`, and `full_temporal` with `temporal_only`, isolates what full-network structure adds for the fraud class.

In the full-graph track, graph features always come from the full graph, but models train on every training-period fraud plus a seeded 10% of legitimate training rows (`models.full_train_legitimate_fraction`), the same rows for every feature set. Thresholds are still chosen on the full validation period and metrics are measured on the full test period, so reported scores reflect the real fraud rate. The full graph build takes a few minutes and roughly 5–7 GB of memory.

Graph model inputs are in/out degree, amount-weighted degree, unique counterparties, and an unseen-account flag, computed from a causal prefix of earlier edges for training rows and from the frozen training graph for validation and test rows. Sparse weighted PageRank and weakly connected components are computed for analysis and shown in the demo, but they are not model inputs: training rows cannot receive them without future edges, so they would be constant during training and vary only at test time.

All precision, recall, and F1 values are fraud-class (minority-class) scores: `isFraud = 1` is the positive class, and majority-class, macro-, and weighted-average scores are never reported. With a 0.13% fraud rate, including the legitimate class would make every model look near-perfect. PR-AUC is likewise computed for the fraud class, and `main_results.csv` also records fraud support and true-positive, false-positive, and false-negative counts.

The notebooks in `code/notebooks/` are exploration entry points; reusable logic stays in `code/src/`. Open them with the repository root as the working directory, or from the notebook directory.

## Demo

After the pipeline has run, start the interactive app:

```powershell
streamlit run code/demo/app.py
```

The app runs offline from cached results and has four sections: the accuracy trap (why only fraud-class metrics are reported), fraud-class results for both tracks, full graph versus sample gains, and a transaction explorer that shows held-out test transactions with every model's score and the accounts' training-graph neighbourhood. Open a section directly with `?section=1` to `?section=4`.

## Presentations

The [project presentation](presentations/Graph_Based_Fraud_Detection.pptx) is from the October milestone. The [demo presentation](presentations/Group19-Project19-FA26-Group-DEMO-Presentation.pptx) is generated from the measured results and demo screenshots:

```powershell
python code/scripts/create_presentation.py
```

See [the project guide](docs/PROJECT_GUIDE.md) for feature definitions, leakage controls, result interpretation, and known limitations.

## Methodology and artifacts

<!-- BEGIN MEASURED RESULTS -->
## Measured experiment results

**Status:** All 12 local custom-graph model/feature experiments completed.

- Dataset: 6,362,620 transactions; 8,213 fraud (0.1291%).
- Chronological boundaries by unique step: train through 520, validation through 631, test through 743. Rows: 6,082,007 train / 191,147 validation / 89,466 test.
- Reported PR-AUC, ROC-AUC, precision, recall, and F1 are test-set measurements. Threshold-dependent metrics use thresholds chosen on validation fraud-class F1 only.
- Precision, recall, and F1 are computed for the minority fraud class only (`isFraud = 1`; scikit-learn `average="binary"`, `pos_label=1`). Majority-class, macro-averaged, and weighted-averaged scores are never reported, because the legitimate class would mask fraud-class performance at a 0.13% fraud rate.
- Sampled custom graph comparisons use the same deterministic sampled training rows for all models and feature sets. Validation and test rows remain unchanged.

| Model | Features | PR-AUC | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic | tabular_sampled | 0.807394 | 0.830059 | 0.674920 | 0.744493 | 0.985130 |
| Logistic | sampled_graph | 0.765801 | 0.832599 | 0.603834 | 0.700000 | 0.979014 |
| Logistic | sampled_temporal | 0.696023 | 0.765890 | 0.577476 | 0.658470 | 0.975122 |
| Random Forest | tabular_sampled | 0.997417 | 0.997590 | 0.992013 | 0.994794 | 0.999934 |
| Random Forest | sampled_graph | 0.997285 | 0.995997 | 0.993610 | 0.994802 | 0.999864 |
| Random Forest | sampled_temporal | 0.997320 | 0.993605 | 0.992812 | 0.993208 | 0.999874 |
| XGBoost | tabular_sampled | 0.969272 | 0.899845 | 0.925719 | 0.912598 | 0.999272 |
| XGBoost | sampled_graph | 0.969146 | 0.901931 | 0.932907 | 0.917157 | 0.999376 |
| XGBoost | sampled_temporal | 0.970173 | 0.911330 | 0.886581 | 0.898785 | 0.999248 |
| MLP | tabular_sampled | 0.751198 | 0.862262 | 0.615016 | 0.717949 | 0.975128 |
| MLP | sampled_graph | 0.741389 | 0.898477 | 0.565495 | 0.694118 | 0.970811 |
| MLP | sampled_temporal | 0.793456 | 0.914498 | 0.589457 | 0.716853 | 0.976450 |

The local custom graph is built from a deterministic random sample of 50,000 training transactions, including 51 fraud rows and 98,145 accounts (seed 42).

All models are fit on these sampled training rows; the three feature sets are tabular-only, tabular plus sampled graph, and tabular plus sampled graph and 24-hour temporal features. Validation and test rows remain unchanged. Graph model inputs are in/out degree, weighted degree, unique counterparties, and an unseen-account flag. Sparse weighted PageRank and weakly connected components are computed for analysis but are not model inputs, because training rows cannot receive them without future edges.

### Full-graph experiments (every training edge)

**Status:** All 16 full-graph model/feature experiments completed.

The full graph uses all 6,082,007 training transactions, including 5,781 fraud rows, 8,669,206 accounts, and 2,587,199 weakly connected components. Every model is fit on all training-period fraud rows plus a seeded subsample of legitimate training rows (`models.full_train_legitimate_fraction`), the same rows for every feature set; validation and test are the same full chronological partitions as above. PageRank and component IDs are not model inputs, because training rows cannot receive them without future edges. Feature sets: `tabular`, `full_graph` (tabular plus full graph), `temporal_only` (tabular plus 24-hour temporal), and `full_temporal` (tabular plus full graph and 24-hour temporal).

| Model | Features | PR-AUC | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic | tabular | 0.762753 | 0.840000 | 0.603834 | 0.702602 | 0.979536 |
| Logistic | full_graph | 0.769156 | 0.858931 | 0.603035 | 0.708588 | 0.979235 |
| Logistic | temporal_only | 0.689653 | 0.787112 | 0.546326 | 0.644979 | 0.975949 |
| Logistic | full_temporal | 0.637326 | 0.753216 | 0.514377 | 0.611296 | 0.969418 |
| Random Forest | tabular | 0.999999 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| Random Forest | full_graph | 1.000000 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| Random Forest | temporal_only | 0.999999 | 1.000000 | 0.999201 | 0.999600 | 1.000000 |
| Random Forest | full_temporal | 0.999997 | 1.000000 | 0.996006 | 0.997999 | 1.000000 |
| XGBoost | tabular | 0.998751 | 0.989715 | 0.999201 | 0.994436 | 0.999984 |
| XGBoost | full_graph | 0.998804 | 0.989715 | 0.999201 | 0.994436 | 0.999984 |
| XGBoost | temporal_only | 0.998799 | 0.989715 | 0.999201 | 0.994436 | 0.999984 |
| XGBoost | full_temporal | 0.998842 | 0.988933 | 0.999201 | 0.994041 | 0.999985 |
| MLP | tabular | 0.941210 | 0.724181 | 0.918530 | 0.809859 | 0.998163 |
| MLP | full_graph | 0.932091 | 0.731043 | 0.900958 | 0.807156 | 0.997634 |
| MLP | temporal_only | 0.887328 | 0.786260 | 0.822684 | 0.804059 | 0.995494 |
| MLP | full_temporal | 0.910001 | 0.690088 | 0.878594 | 0.773015 | 0.996550 |

Fraud-class change from adding full-graph features (treatment minus baseline):

| Model | Comparison | PR-AUC | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---:|---:|---:|---:|
| Logistic | `full_graph` minus `tabular` | +0.006403 | +0.018931 | -0.000799 | +0.005985 |
| Random Forest | `full_graph` minus `tabular` | +0.000001 | +0.000000 | +0.000000 | +0.000000 |
| XGBoost | `full_graph` minus `tabular` | +0.000054 | +0.000000 | +0.000000 | +0.000000 |
| MLP | `full_graph` minus `tabular` | -0.009118 | +0.006862 | -0.017572 | -0.002704 |
| Logistic | `full_temporal` minus `temporal_only` | -0.052328 | -0.033895 | -0.031949 | -0.033683 |
| Random Forest | `full_temporal` minus `temporal_only` | -0.000003 | +0.000000 | -0.003195 | -0.001601 |
| XGBoost | `full_temporal` minus `temporal_only` | +0.000043 | -0.000782 | +0.000000 | -0.000395 |
| MLP | `full_temporal` minus `temporal_only` | +0.022673 | -0.096172 | +0.055911 | -0.031045 |

The PaySim CSV was obtained from [Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. The raw dataset is not included in this repository.
<!-- END MEASURED RESULTS -->

- The train/validation/test split is based on ordered unique `step` values and stored in `data/processed/split.json`. All experiments use the same untouched chronological validation and test partitions.
- Tabular predictors exclude account IDs, `isFraud`, and `isFlaggedFraud`. Scaling is fit only on training rows.
- Temporal features use the half-open interval `[step - window, step)`. All transactions at a step receive features before any transaction from that step updates history.
- The production temporal feature command uses DuckDB window queries with bounded memory and disk spill, rather than retaining Python account histories for the full dataset. Its numeric `RANGE BETWEEN window PRECEDING AND 1 PRECEDING` frames represent `[step-window, step)`, excluding all transactions in the current hourly step.
- Graph edges are taken from rows at or before `train_end_step` (all of them for the full graph, a seeded sample of 50,000 for the sampled graph). During training, degree, weighted-degree, counterparty, and unseen-account values are computed from a causal prefix of edges; no same-step or future edges enter those features. Validation and test graph features are computed from the frozen training graph. PageRank and component IDs are excluded from model inputs because global graph summaries would expose later training edges to earlier transactions.
- Positive class weights are calculated from the rows each model trains on. Validation PR-AUC early-stops the MLP. Thresholds are selected on validation fraud F1 and then frozen for test evaluation.
- Feature caches are Parquet under `data/features/`; splits are under `data/processed/`. Main test metrics are written to `evaluations/metrics/main_results.csv`, figures to `evaluations/figures/`, and per-model test predictions to `evaluations/predictions/` (not committed).
- The experiment run also writes `evaluations/metrics/research_summary.md`, which answers the research questions using measured test results, and `evaluations/metrics/fraud_class_gains.csv` with per-model fraud-class PR-AUC, precision, recall, and F1 changes between feature sets. ROC-AUC and F1 are undefined when validation or test lacks one class; in that case the experiment stops with a clear error instead of publishing misleading metrics.
- Raw data, feature caches, and per-row predictions are ignored by Git; metrics, figures, and summaries in `evaluations/` are committed.

Run tests with:

```powershell
python -m pytest code/tests
```
