from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions
from src.graph.static_features import attach_account_features, build_training_graph
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    transactions = load_transactions(config["data"]["raw_path"])
    split_path = Path("data/processed/split.json")
    if not split_path.is_file():
        raise FileNotFoundError("Run scripts/run_preprocessing.py before graph features")
    boundaries = json.loads(split_path.read_text(encoding="utf-8"))
    feature_dir = Path("data/features")
    feature_dir.mkdir(parents=True, exist_ok=True)

    graph_config = config["graph"]
    password = os.environ.get(graph_config["password_env"])
    if not password:
        raise RuntimeError(
            f"Set {graph_config['password_env']} for the configured Neo4j database"
        )
    driver = GraphDatabase.driver(
        graph_config["uri"],
        auth=(graph_config["username"], password),
    )
    try:
        driver.verify_connectivity()
        account_features = build_training_graph(
            driver,
            transactions,
            int(boundaries["train_end_step"]),
            database=graph_config["database"],
            batch_size=int(graph_config["batch_size"]),
            pagerank_iterations=int(graph_config["pagerank_iterations"]),
        )
    finally:
        driver.close()
    static_features = attach_account_features(
        transactions,
        account_features,
        train_end_step=int(boundaries["train_end_step"]),
    )
    static_features.insert(0, "_row_id", range(len(static_features)))
    static_features.to_parquet(feature_dir / "static_graph_features.parquet", index=False)
    account_features.to_parquet(feature_dir / "account_graph_features.parquet", index=False)
    figure_dir = Path("results/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    for column, filename in (
        ("pagerank", "pagerank_distribution.png"),
        ("in_degree", "graph_degree_distribution.png"),
    ):
        axis = account_features[column].clip(
            upper=account_features[column].quantile(0.99)
        ).plot.hist(bins=50, figsize=(8, 5))
        axis.set_title(column.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=160)
        plt.close()
    account_features["community_id"].value_counts().plot.hist(
        bins=50, figsize=(8, 5)
    )
    plt.title("Community size distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "community_size_distribution.png", dpi=160)
    plt.close()

    print(
        f"Saved {len(static_features):,} static graph feature rows; "
        f"training graph cutoff step={boundaries['train_end_step']}."
    )


if __name__ == "__main__":
    main()
