"""Run every pipeline stage in order. Run from the repository root:

    python code/run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.utils.config import load_config

SCRIPTS = Path(__file__).resolve().parent / "scripts"


def main() -> None:
    stages = [
        "run_preprocessing.py",
        "run_temporal_features.py",
        "run_sampled_graph_features.py",
        "run_experiments.py",
    ]
    if load_config()["graph"].get("full_graph", True):
        stages += [
            "run_full_graph_features.py",
            "run_full_graph_experiments.py",
        ]
    stages.append("prepare_demo.py")
    for stage in stages:
        subprocess.run([sys.executable, str(SCRIPTS / stage)], check=True)


if __name__ == "__main__":
    main()
