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
    seed = int(config["seed"])
    metadata = build_graph_feature_cache(
        transactions,
        train_end_step=int(boundaries["train_end_step"]),
        kind="sampled",
        seed=seed,
        edge_limit=int(config["graph"]["sampled_edge_limit"]),
    )
    print(
        f"Saved sampled graph features for {metadata['sampled_training_edges']:,} training "
        f"edges and {metadata['sampled_training_accounts']:,} accounts "
        f"(seed={seed}, cutoff step={boundaries['train_end_step']})."
    )


if __name__ == "__main__":
    main()
