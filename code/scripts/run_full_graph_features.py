from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions
from src.graph.account_graph import build_graph_feature_cache
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    transactions = load_transactions(config["data"]["raw_path"])
    split_path = Path("data/processed/split.json")
    if not split_path.is_file():
        raise FileNotFoundError("Run code/scripts/run_preprocessing.py before graph features")
    boundaries = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = build_graph_feature_cache(
        transactions,
        train_end_step=int(boundaries["train_end_step"]),
        kind="full",
        seed=int(config["seed"]),
    )
    print(
        f"Saved full graph features for {metadata['full_training_edges']:,} training "
        f"edges, {metadata['full_training_accounts']:,} accounts, and "
        f"{metadata['full_weak_components']:,} weakly connected components "
        f"(cutoff step={boundaries['train_end_step']})."
    )


if __name__ == "__main__":
    main()
