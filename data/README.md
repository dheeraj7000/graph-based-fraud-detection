# Data

The project uses **PaySim**, a synthetic mobile-money transaction dataset generated from real transaction logs: [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) on Kaggle (CC BY-SA 4.0).

| Property | Value |
|---|---|
| Transactions | 6,362,620 |
| Fraud transactions | 8,213 (0.129%) |
| Time span | 743 hourly steps (about 31 days) |
| Columns | 11 (`step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`) |
| Raw file | `PS_20174392719_1491204439457_log.csv`, 494 MB |

The full CSV is too large for Git, so it is not committed.

## Getting the full dataset

Download it from Kaggle and place it at the path configured by `data.raw_path` in `config/config.yaml`:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Or download it with `kagglehub` (no Kaggle account needed for public datasets):

```bash
python -c "import kagglehub, shutil, pathlib; src = pathlib.Path(kagglehub.dataset_download('ealaxi/paysim1')); shutil.copy(next(src.glob('*.csv')), 'data/raw/')"
```

## Directory layout

| Folder | Contents | In Git |
|---|---|---|
| `raw/` | The PaySim CSV | No |
| `sample/` | `paysim_sample_20k.csv`: 20,000 rows (1,000 fraud, 19,000 legitimate) drawn with seed 42, for inspecting the schema without the full download | Yes |
| `processed/` | Chronological split metadata, graph metadata, demo cache (generated) | No |
| `features/` | Parquet feature caches: tabular, temporal, sampled graph, full graph (generated) | No |

The sample over-represents fraud (5% instead of 0.129%) so it contains enough positive examples to look at; it is not used by the experiments. It was created with:

```python
df = pd.read_csv("data/raw/PS_20174392719_1491204439457_log.csv")
sample = pd.concat([
    df[df.isFraud == 1].sample(1000, random_state=42),
    df[df.isFraud == 0].sample(19000, random_state=42),
]).sort_index()
```

## Preprocessing

`code/scripts/run_preprocessing.py` loads the CSV with explicit types and rejects missing columns, missing values, or non-binary labels. It builds 17 ID-free tabular features (amount, balances, balance changes, amount-to-balance ratios, zero-balance flags, one-hot transaction type) and splits the data chronologically by `step` into train (steps 1–520), validation (521–631), and test (632–743). Account IDs are used only to build the account graph and 24-hour activity features; `isFlaggedFraud` is never used as a predictor.

## Training rows per track

| Track | Graph built from | Model training rows | Validation and test |
|---|---|---|---|
| Sampled | 50,000 seeded training edges (98,145 accounts) | The same 50,000 rows (51 frauds) | Full |
| Full graph | All 6,082,007 training edges (8,669,206 accounts) | All 5,781 training frauds plus a seeded 10% of legitimate training rows | Full |

Within each track, every feature set trains on exactly the same rows, so feature-set differences are not confounded with sampling.
