# Project Guide: Graph-Based Fraud Detection

## Project at a glance

This project studies whether account-network and recent-activity features help identify fraudulent transactions in PaySim. It compares four conventional classifiers across three feature sets, with a chronological split and a resource-bounded local graph implementation.

The graph is a feature-engineering step, not a graph neural network. The project does not require Neo4j, AuraDB, or another graph database service.

## Data and evaluation design

The experiments use the PaySim synthetic financial transaction dataset: 6,362,620 transactions, including 8,213 fraud cases (about 0.1291%). Its strong class imbalance makes precision-recall area (PR-AUC) more informative than accuracy as the primary ranking metric.

| Partition | Rows | Step range |
|---|---:|---|
| Train | 6,082,007 | Through step 520 |
| Validation | 191,147 | Steps 521–631 |
| Test | 89,466 | Steps 632–743 |

Partitions are chronological by unique transaction step. The validation and test periods are not downsampled. For the matched feature comparison, each model is fitted on the same deterministic sample of 50,000 training transactions (seed 42), containing 51 fraud transactions and 98,145 distinct accounts. That sample is about 0.82% of the training rows. A second, full-graph track builds its graph from every training transaction (see below).

## Feature sets and experiment matrix

The sampled comparison is a 4-model × 3-feature-set matrix:

| Feature set | Inputs |
|---|---|
| `tabular_sampled` | Transaction-level numeric predictors; fitted on the sampled training rows |
| `sampled_graph` | Tabular predictors plus local account-network features |
| `sampled_temporal` | Tabular, sampled-graph, and 24-hour causal temporal features |

The four model families are Logistic Regression, Random Forest, XGBoost, and a multilayer perceptron (MLP).

### Full-graph track

The full-graph track tests whether full-network structural features help the minority fraud class once the graph is no longer limited to a 50,000-edge sample. It builds the account graph from all 6,082,007 training-period edges (8,669,206 accounts). Models train on every training-period fraud (5,781) plus a seeded 10% of legitimate training rows (`models.full_train_legitimate_fraction`), the same rows for every feature set; validation and test stay full. This gives a 4-model × 4-feature-set matrix:

| Feature set | Inputs |
|---|---|
| `tabular` | Transaction-level numeric predictors |
| `full_graph` | Tabular predictors plus full-training-graph account features |
| `temporal_only` | Tabular plus 24-hour causal temporal features |
| `full_temporal` | Tabular, full-graph, and 24-hour causal temporal features |

`full_graph` minus `tabular` measures the full-network contribution; `full_temporal` minus `temporal_only` measures it on top of recent activity. Both differences are reported per model for fraud-class PR-AUC, precision, recall, and F1. Run it with `python code/scripts/run_full_graph_features.py` followed by `python code/scripts/run_full_graph_experiments.py`.

### Graph and temporal features

Transactions are directed, weighted edges from the origin account to the destination account; transaction amount is the edge weight. The account graph is held as a SciPy sparse matrix, which provides weighted PageRank (power iteration with the same weighting and dangling-node handling as NetworkX) and weakly connected component IDs; degree, weighted-degree, and unique-counterparty counts are vectorized NumPy aggregates. The same code builds the 50,000-edge sample and the full training graph. These account summaries are joined to each transaction's source and destination.

Training-row degree, weighted-degree, counterparty, and unseen-account features use a causal prefix computed with DuckDB window queries: transactions at a step receive features before any transaction from that step updates the history. Global PageRank and component values cannot be assigned to training rows without exposing later training edges to earlier observations, so they would be constant during training and vary only at test time. They are therefore computed for analysis and shown in the demo, but excluded from model inputs. Validation and test graph features are derived from the frozen training graph.

Temporal aggregates use the half-open window `[step - window, step)`. Thus, all transactions in the current hourly step are excluded from that step's features. The primary matrix uses a 24-hour window; the pipeline also writes 6-hour and 72-hour sensitivity caches.

## Training and metrics

- Scaling and model fitting use training data only.
- Class weighting is derived from the rows each model trains on.
- MLP early stopping uses validation PR-AUC.
- A decision threshold is selected on validation fraud-class F1 and then frozen for the test period.
- Test reporting includes PR-AUC, ROC-AUC, precision, recall, and F1. PR-AUC is the primary ranking metric; the threshold-dependent measures depend on the validation-selected operating point.
- **Precision, recall, and F1 are reported for the minority fraud class only** (`isFraud = 1`, scikit-learn `average="binary"`, `pos_label=1`). Majority-class, macro-averaged, and weighted-averaged scores are never reported: at a 0.13% fraud rate the legitimate class would dominate them and mask fraud-detection performance. Results also record the test fraud support and true-positive, false-positive, and false-negative counts.

## Measured main-matrix results

All values below are measured on the untouched chronological test split (89,466 transactions, 1,252 frauds). Scores are fraud-class F1 at the validation-selected threshold.

| Model | Sampled tabular | Sampled + graph | Sampled + graph + 24h | Full tabular | Full + graph | Full + 24h | Full + graph + 24h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7445 | 0.7000 | 0.6585 | 0.7026 | 0.7086 | 0.6450 | 0.6113 |
| Random Forest | 0.9948 | 0.9948 | 0.9932 | **0.9996** | **0.9996** | **0.9996** | 0.9980 |
| XGBoost | 0.9126 | 0.9172 | 0.8988 | 0.9944 | 0.9944 | 0.9944 | 0.9940 |
| MLP | 0.7179 | 0.6941 | 0.7169 | 0.8099 | 0.8072 | 0.8041 | 0.7730 |

Full-network structural features do not move the needle for the fraud class. Adding the full graph to tabular features changed fraud F1 by +0.0008 on average (only Logistic Regression improved, by +0.0060; Random Forest and XGBoost were unchanged; the MLP fell by 0.0027), and adding it on top of 24-hour activity lowered fraud F1 for all four models (mean −0.0167). The 50,000-edge sample gave a similar answer (mean −0.0159). Tabular features alone already reach fraud F1 0.9996 with Random Forest (1,251 of 1,252 test frauds caught, no false alarms) and 0.9944 with XGBoost, leaving little room for any added block. PR-AUC shows the same pattern.

Per-model fraud-class precision, recall, F1, ROC-AUC, thresholds, and confusion matrices are published in `evaluations/metrics/main_results.csv` when generated. Supplemental and full-graph measurements are published in the README; per-model fraud-class gains are in `evaluations/metrics/fraud_class_gains.csv`.

## Interpretation and limitations

1. **Synthetic benchmark:** PaySim is simulated data. Near-perfect tree-model scores should not be read as evidence of equivalent performance on live financial transactions.
2. **Sample scope:** The sampled matrix uses 50,000 training edges, not all 6.08 million training transactions. The full-graph track removes that limit; when comparing the two, remember they also differ in training-set size, so compare each graph gain with its own matched baseline.
3. **Sampling variability:** A single seed and sample do not quantify uncertainty. Repeat with multiple seeds or stratified sampling before drawing robust comparative conclusions.
4. **Temporal scope:** The evaluation uses one chronological train/validation/test division. Performance can shift under other time periods or distribution changes.
5. **No GNN:** MLP is a feed-forward classifier, not a graph neural network. Graph structure enters through engineered features.
6. **Threshold dependence:** Precision, recall, and F1 use a threshold selected for validation F1; they are not threshold-free ranking measures.
7. **Full-track training subsample:** Full-graph models train on every training fraud but only 10% of legitimate training rows. Graph features and evaluation still use all data, and class weights follow the subsample.
8. **Privacy and data availability:** Raw PaySim data, feature caches, and per-row predictions are not committed. Obtain the dataset from its [Kaggle source](https://www.kaggle.com/datasets/ealaxi/paysim1) (CC BY-SA 4.0).

## Reproduce the workflow

Use Python 3.11 or newer. Install dependencies and place the PaySim CSV at the configured `data.raw_path`, then run:

```powershell
python -m pip install -r requirements.txt
python code/scripts/run_preprocessing.py
python code/scripts/run_temporal_features.py
python code/scripts/run_sampled_graph_features.py
python code/scripts/run_experiments.py
python code/scripts/run_full_graph_features.py
python code/scripts/run_full_graph_experiments.py
python code/scripts/create_presentation.py
```

`python code/run_pipeline.py` runs preprocessing, temporal features, sampled graph features, sampled experiments, (unless `graph.full_graph` is `false`) the full-graph features and experiments, and the demo cache in sequence. It does not automatically regenerate the presentation.

## Repository map

| Path | Purpose |
|---|---|
| `config/config.yaml` | Dataset path, sample cap, full-graph toggle, training subsample fraction, temporal windows, and model settings |
| `code/src/data/` | Loading and preprocessing |
| `code/src/features/` | Tabular and temporal feature generation |
| `code/src/graph/` | Sampled and full sparse account graphs and causal graph features |
| `code/src/evaluation/` | Model orchestration, metrics, and publication |
| `code/scripts/` | Reproducible pipeline entry points and presentation generator |
| `code/demo/` | Streamlit demo app |
| `code/tests/` | Pipeline and leakage-invariant tests |
| `data/` | Dataset description and a 20,000-row sample (raw data and caches are not committed) |
| `evaluations/` | Measured metrics, figures, and research summary (per-row predictions are not committed) |
| `presentations/` | Project and demo presentations |

Run the test suite with `python -m pytest code/tests`.
