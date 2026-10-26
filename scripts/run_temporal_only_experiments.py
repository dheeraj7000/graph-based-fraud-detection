from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.experiments import run_experiments


if __name__ == "__main__":
    run_experiments(feature_groups=("temporal_only",))
