from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions
from src.graph.sampled_features import (
    attach_sampled_graph_features,
    sample_training_edges,
    sampled_graph_account_features,
)
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    transactions = load_transactions(config["data"]["raw_path"])
    split_path = Path("data/processed/split.json")
    if not split_path.is_file():
        raise FileNotFoundError("Run scripts/run_preprocessing.py before graph features")
    boundaries = json.loads(split_path.read_text(encoding="utf-8"))
    graph_config = config["graph"]
    edge_limit = int(graph_config["sampled_edge_limit"])
    seed = int(config["seed"])

    sampled_training = sample_training_edges(
        transactions,
        train_end_step=int(boundaries["train_end_step"]),
        max_edges=edge_limit,
        seed=seed,
    )
    account_features = sampled_graph_account_features(sampled_training)
    row_features = attach_sampled_graph_features(
        transactions,
        sampled_training,
        account_features,
        train_end_step=int(boundaries["train_end_step"]),
    )
    feature_dir = Path("data/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "sampled_graph_features.parquet"
    row_features.insert(0, "_row_id", transactions.index.astype("int64"))
    row_features.to_parquet(feature_path, index=False)
    account_features.to_parquet(
        feature_dir / "sampled_graph_account_features.parquet", index=False
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"_row_id": sampled_training.index.astype("int64")}).to_parquet(
        "data/processed/sampled_graph_train_rows.parquet", index=False
    )
    metadata = {
        "backend": "NetworkX",
        "seed": seed,
        "edge_limit": edge_limit,
        "sampled_training_edges": len(sampled_training),
        "sampled_training_frauds": int(sampled_training["isFraud"].sum()),
        "sampled_training_accounts": len(account_features),
        "train_end_step": int(boundaries["train_end_step"]),
    }
    Path("data/processed/sampled_graph_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved sampled graph features for {len(sampled_training):,} training "
        f"edges and {len(account_features):,} accounts "
        f"(seed={seed}, cutoff step={boundaries['train_end_step']})."
    )


if __name__ == "__main__":
    main()
