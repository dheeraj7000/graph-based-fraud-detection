from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.temporal_features import stream_temporal_features
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    feature_dir = Path("data/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(config["temporal"]["primary_window"])]
    windows.extend(int(value) for value in config["temporal"].get("sensitivity_windows", []))
    for window in dict.fromkeys(windows):
        output = (
            feature_dir / "temporal_graph_features.parquet"
            if window == int(config["temporal"]["primary_window"])
            else feature_dir / f"temporal_graph_features_{window}h.parquet"
        )
        row_count = stream_temporal_features(
            config["data"]["raw_path"],
            output,
            window=window,
        )
        print(f"Saved {row_count:,} rolling feature rows for {window}h.")


if __name__ == "__main__":
    main()
