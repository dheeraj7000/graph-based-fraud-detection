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

Partitions are chronological by unique transaction step. The validation and test periods are not downsampled. For the matched feature comparison, each model is fitted on the same deterministic sample of 50,000 training transactions (seed 42), containing 51 fraud transactions and 98,145 distinct accounts. That sample is about 0.82% of the training rows.

## Feature sets and experiment matrix

The main comparison is a 4-model × 3-feature-set matrix:

| Feature set | Inputs |
|---|---|
| `tabular_sampled` | Transaction-level numeric predictors; fitted on the sampled training rows |
| `sampled_graph` | Tabular predictors plus local account-network features |
| `sampled_temporal` | Tabular, sampled-graph, and 24-hour causal temporal features |

The four model families are Logistic Regression, Random Forest, XGBoost, and a multilayer perceptron (MLP). The full-training tabular baselines and the separate temporal-only results in the README are supplemental, not part of this matched matrix.

### Graph and temporal features

Transactions are directed, weighted edges from the origin account to the destination account; transaction amount is the edge weight. NetworkX computes account in/out degree, weighted degree, unique-counterparty counts, weighted PageRank, and weakly connected component IDs. These account summaries are joined to each transaction's source and destination.

Training-row degree, weighted-degree, counterparty, and unseen-account features use a causal prefix: transactions at a step receive features before any transaction from that step updates the history. Global PageRank and component values are not assigned to training rows, because they would expose later training edges to earlier observations. Validation and test graph features are derived from the frozen sampled training graph.

Temporal aggregates use the half-open window `[step - window, step)`. Thus, all transactions in the current hourly step are excluded from that step's features. The primary matrix uses a 24-hour window; the pipeline also writes 6-hour and 72-hour sensitivity caches.

## Training and metrics

- Scaling and model fitting use training data only.
- Class weighting is derived from the training split.
- MLP early stopping uses validation PR-AUC.
- A decision threshold is selected on validation F1 and then frozen for the test period.
- Test reporting includes PR-AUC, ROC-AUC, precision, recall, and F1. PR-AUC is the primary ranking metric; the threshold-dependent measures depend on the validation-selected operating point.

## Measured main-matrix results

All values below are measured on the untouched chronological test split. Scores are PR-AUC.

| Model | Sampled tabular | Sampled graph | Sampled graph + 24h |
|---|---:|---:|---:|
| Logistic Regression | 0.807399 | 0.765795 | 0.696202 |
| Random Forest | 0.997418 | 0.996614 | **0.998459** |
| XGBoost | 0.968646 | 0.970304 | 0.971133 |
| MLP | 0.740390 | 0.751701 | 0.796303 |

The highest measured main-matrix PR-AUC is 0.998459 for Random Forest with sampled graph and temporal features. This does not mean graph features improve every model: adding graph features alone improved PR-AUC for two of four model families, and the mean graph-minus-tabular change was -0.007360. Adding 24-hour temporal features to the sampled-graph set improved three of four model families, while the mean change across models was -0.005579. Logistic Regression accounts for a large part of those negative averages; the MLP saw the largest overall gain from sampled tabular to sampled graph plus temporal (+0.055913).

Per-model precision, recall, F1, ROC-AUC, thresholds, and confusion matrices are published in `results/metrics/main_results.csv` when generated. Supplemental measurements are published in the README.

## Interpretation and limitations

1. **Synthetic benchmark:** PaySim is simulated data. Near-perfect tree-model scores should not be read as evidence of equivalent performance on live financial transactions.
2. **Sample scope:** The graph contains 50,000 sampled training edges, not all 6.08 million training transactions. Results describe this deterministic sampled setup; they are not full-graph estimates.
3. **Sampling variability:** A single seed and sample do not quantify uncertainty. Repeat with multiple seeds or stratified sampling before drawing robust comparative conclusions.
4. **Temporal scope:** The evaluation uses one chronological train/validation/test division. Performance can shift under other time periods or distribution changes.
5. **No GNN:** MLP is a feed-forward classifier, not a graph neural network. Graph structure enters through engineered features.
6. **Threshold dependence:** Precision, recall, and F1 use a threshold selected for validation F1; they are not threshold-free ranking measures.
7. **Privacy and data availability:** Raw PaySim data and generated caches/results are not committed. Obtain the dataset from its [Kaggle source](https://www.kaggle.com/datasets/ealaxi/paysim1) (CC BY-SA 4.0).

## Reproduce the workflow

Use Python 3.11 or newer. Install dependencies and place the PaySim CSV at the configured `data.raw_path`, then run:

```powershell
python -m pip install -r requirements.txt
python scripts/run_preprocessing.py
python scripts/run_temporal_features.py
python scripts/run_sampled_graph_features.py
python scripts/run_experiments.py
python scripts/create_presentation.py
```

`python run_pipeline.py` runs preprocessing, temporal features, sampled graph features, and experiments in sequence. It does not automatically regenerate the presentation.

## Repository map

| Path | Purpose |
|---|---|
| `config/config.yaml` | Dataset path, sample cap, temporal windows, and model settings |
| `src/data/` | Loading and preprocessing |
| `src/features/` | Tabular and temporal feature generation |
| `src/graph/` | Deterministic NetworkX sample and causal graph features |
| `src/evaluation/` | Model orchestration, metrics, and publication |
| `scripts/` | Reproducible pipeline entry points and presentation generator |
| `results/metrics/` | Generated scores and summaries (ignored by Git) |
| `presentations/` | Generated PowerPoint deliverable |
| `tests/` | Pipeline and leakage-invariant tests |

Run the test suite with `python -m pytest`.
