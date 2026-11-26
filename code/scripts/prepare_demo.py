"""Build the small cache that powers the Streamlit demo (code/demo/app.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions
from src.evaluation.experiments import (
    FULL_FEATURE_GROUPS,
    MODEL_ORDER,
    SAMPLED_FEATURE_GROUPS,
)
from src.graph.account_graph import GRAPH_COLUMNS
from src.utils.config import load_config

DEMO_DIR = Path("data/processed/demo")
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]
LEGITIMATE_SAMPLE = 3000
MAX_NEIGHBOR_EDGES = 400_000


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    split = json.loads(Path("data/processed/split.json").read_text(encoding="utf-8"))
    train_end_step = int(split["train_end_step"])
    metadata = pd.read_parquet("data/processed/split_metadata.parquet")
    test_ids = metadata.loc[metadata["split"] == "test", "_row_id"].to_numpy()

    transactions = load_transactions(config["data"]["raw_path"])
    test = transactions.loc[test_ids, RAW_COLUMNS].copy()
    test.insert(0, "_row_id", test_ids)
    for column in ("type", "nameOrig", "nameDest"):
        test[column] = test[column].astype(str)

    thresholds = []
    for model in MODEL_ORDER:
        for group in (*FULL_FEATURE_GROUPS, *SAMPLED_FEATURE_GROUPS):
            path = Path("evaluations/predictions") / f"{model}_{group}.parquet"
            if not path.is_file():
                continue
            predictions = pd.read_parquet(path)
            if not np.array_equal(predictions["_row_id"].to_numpy(), test_ids):
                raise ValueError(f"{path} rows do not match the test partition")
            test[f"p::{model}::{group}"] = predictions["probability"].to_numpy(
                dtype=np.float32
            )
            thresholds.append(
                {
                    "model": model,
                    "features": group,
                    "threshold": float(predictions["threshold"].iloc[0]),
                }
            )
    if not thresholds:
        raise FileNotFoundError("No prediction files found; run the experiments first")

    probability_columns = [column for column in test.columns if column.startswith("p::")]
    threshold_lookup = {
        f"p::{row['model']}::{row['features']}": row["threshold"] for row in thresholds
    }
    flagged_anywhere = np.zeros(len(test), dtype=bool)
    for column in probability_columns:
        flagged_anywhere |= test[column].to_numpy() >= threshold_lookup[column]
    fraud = test["isFraud"].to_numpy() == 1
    legitimate_pool = np.flatnonzero(~fraud & ~flagged_anywhere)
    rng = np.random.default_rng(seed)
    sampled_legitimate = rng.choice(
        legitimate_pool, size=min(LEGITIMATE_SAMPLE, len(legitimate_pool)), replace=False
    )
    keep = np.zeros(len(test), dtype=bool)
    keep[fraud | flagged_anywhere] = True
    keep[sampled_legitimate] = True
    demo = test.loc[keep].reset_index(drop=True)

    full_graph_path = Path("data/features/full_graph_features.parquet")
    if full_graph_path.is_file():
        graph = pd.read_parquet(full_graph_path).set_index("_row_id")
        graph_values = graph.loc[demo["_row_id"], GRAPH_COLUMNS].reset_index(drop=True)
        demo = pd.concat([demo, graph_values.add_prefix("g::")], axis=1)

    training = transactions.loc[
        transactions["step"] <= train_end_step,
        ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
    ]
    focus_accounts = pd.Index(pd.unique(demo[["nameOrig", "nameDest"]].to_numpy().ravel()))
    origin = training["nameOrig"].astype(str)
    destination = training["nameDest"].astype(str)
    first_hop = origin.isin(focus_accounts) | destination.isin(focus_accounts)
    neighbors = pd.Index(
        pd.unique(np.concatenate([origin[first_hop], destination[first_hop]]))
    ).difference(focus_accounts)
    second_hop = ~first_hop & (origin.isin(neighbors) | destination.isin(neighbors))
    second_hop_rows = np.flatnonzero(second_hop.to_numpy())
    if len(second_hop_rows) > MAX_NEIGHBOR_EDGES:
        second_hop_rows = np.sort(
            rng.choice(second_hop_rows, size=MAX_NEIGHBOR_EDGES, replace=False)
        )
    edge_mask = first_hop.to_numpy().copy()
    edge_mask[second_hop_rows] = True
    edges = training.loc[edge_mask].copy()
    for column in ("type", "nameOrig", "nameDest"):
        edges[column] = edges[column].astype(str)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    demo.to_parquet(DEMO_DIR / "demo_transactions.parquet", index=False)
    pd.DataFrame(thresholds).to_parquet(DEMO_DIR / "thresholds.parquet", index=False)
    edges.reset_index(drop=True).to_parquet(DEMO_DIR / "ego_edges.parquet", index=False)
    print(
        f"Saved demo cache: {len(demo):,} test transactions "
        f"({int(demo['isFraud'].sum()):,} fraud) and {len(edges):,} training-graph edges "
        f"to {DEMO_DIR}"
    )


if __name__ == "__main__":
    main()
