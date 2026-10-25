from __future__ import annotations

import subprocess
import sys


def main() -> None:
    stages = (
        "scripts/run_preprocessing.py",
        "scripts/run_temporal_features.py",
        "scripts/run_graph_features.py",
        "scripts/run_experiments.py",
    )
    for stage in stages:
        subprocess.run([sys.executable, stage], check=True)


if __name__ == "__main__":
    main()
