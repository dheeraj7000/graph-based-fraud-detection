from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> float:
    if len(y_validation) != len(probabilities) or len(y_validation) == 0:
        raise ValueError("Validation labels and probabilities must be non-empty and aligned")
    if np.unique(y_validation).size != 2:
        raise ValueError(
            "Validation labels must contain both classes to select an F1 threshold"
        )
    precision, recall, thresholds = precision_recall_curve(
        y_validation, probabilities
    )
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | list[list[int]]]:
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must be aligned")
    if np.unique(y_true).size != 2:
        raise ValueError("Test labels must contain both classes to calculate ROC-AUC")
    predicted = probabilities >= threshold
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def evaluate_with_validation_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[float, dict[str, float | list[list[int]]]]:
    threshold = select_threshold(y_validation, validation_probabilities)
    return threshold, evaluate_predictions(y_test, test_probabilities, threshold)
