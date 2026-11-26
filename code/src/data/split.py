from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalBoundaries:
    train_end_step: int
    validation_end_step: int
    test_end_step: int


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    step_column: str = "step",
) -> tuple[pd.Series, pd.Series, pd.Series, ChronologicalBoundaries]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive and sum to 1")
    if step_column not in frame:
        raise ValueError(f"Missing chronological column: {step_column}")

    steps = pd.Index(frame[step_column].dropna().unique()).sort_values()
    if len(steps) < 3:
        raise ValueError("At least three distinct steps are required for splitting")
    train_end = max(1, int(len(steps) * train_ratio))
    validation_end = max(train_end + 1, int(len(steps) * (train_ratio + validation_ratio)))
    if validation_end >= len(steps):
        validation_end = len(steps) - 1
    train_end_step = int(steps[train_end - 1])
    validation_end_step = int(steps[validation_end - 1])
    test_end_step = int(steps[-1])
    step_values = frame[step_column]
    train_mask = step_values <= train_end_step
    validation_mask = (step_values > train_end_step) & (
        step_values <= validation_end_step
    )
    test_mask = step_values > validation_end_step
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Chronological split produced an empty partition")
    boundaries = ChronologicalBoundaries(
        train_end_step=train_end_step,
        validation_end_step=validation_end_step,
        test_end_step=test_end_step,
    )
    return train_mask, validation_mask, test_mask, boundaries


def boundaries_as_dict(boundaries: ChronologicalBoundaries) -> dict[str, int]:
    return asdict(boundaries)
