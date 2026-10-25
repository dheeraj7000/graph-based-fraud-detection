from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions, make_tabular_features
from src.data.split import boundaries_as_dict, chronological_split
from src.evaluation.eda import generate_eda
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    data_path = Path(config["data"]["raw_path"])
    transactions = load_transactions(data_path)
    feature_dir = Path("data/features")
    processed_dir = Path("data/processed")
    metric_dir = Path("results/metrics")
    feature_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    tabular = make_tabular_features(transactions)
    train, validation, test, boundaries = chronological_split(
        transactions,
        config["data"]["train_ratio"],
        config["data"]["validation_ratio"],
        config["data"]["test_ratio"],
    )
    tabular.insert(0, "_row_id", range(len(tabular)))
    tabular.to_parquet(feature_dir / "tabular_features.parquet", index=False)
    metadata = pd.DataFrame(
        {
            "_row_id": range(len(transactions)),
            "step": transactions["step"].to_numpy(),
            "isFraud": transactions["isFraud"].astype("int8").to_numpy(),
            "split": "train",
        }
    )
    metadata.loc[validation, "split"] = "validation"
    metadata.loc[test, "split"] = "test"
    metadata.to_parquet(processed_dir / "split_metadata.parquet", index=False)
    split_config = {
        **boundaries_as_dict(boundaries),
        "train_steps": int(train.sum()),
        "validation_steps": int(validation.sum()),
        "test_steps": int(test.sum()),
        "distinct_steps": int(transactions["step"].nunique()),
    }
    (processed_dir / "split.json").write_text(
        json.dumps(split_config, indent=2), encoding="utf-8"
    )
    report = generate_eda(transactions, "results/figures/eda")
    (metric_dir / "eda_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(split_config, indent=2))


if __name__ == "__main__":
    main()
