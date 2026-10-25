from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import load_transactions
from src.graph.temporal_features import build_temporal_features
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    transactions = load_transactions(config["data"]["raw_path"])
    feature_dir = Path("data/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(config["temporal"]["primary_window"])]
    windows.extend(int(value) for value in config["temporal"].get("sensitivity_windows", []))
    for window in dict.fromkeys(windows):
        features = build_temporal_features(transactions, window=window)
        features.insert(0, "_row_id", range(len(features)))
        features.to_parquet(
            feature_dir / f"temporal_graph_features_{window}h.parquet", index=False
        )
        if window == int(config["temporal"]["primary_window"]):
            features.to_parquet(feature_dir / "temporal_graph_features.parquet", index=False)
        print(f"Saved rolling features for {window}h.")


if __name__ == "__main__":
    main()
