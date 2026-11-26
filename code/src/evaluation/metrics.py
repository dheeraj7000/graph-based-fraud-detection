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

FRAUD_LABEL = 1
FRAUD_CLASS_SCORING = {"pos_label": FRAUD_LABEL, "average": "binary", "zero_division": 0}


def _require_binary_labels(labels: np.ndarray, name: str) -> None:
    if not set(np.unique(labels).tolist()).issubset({0, FRAUD_LABEL}):
        raise ValueError(f"{name} labels must be binary 0/1 with 1 meaning fraud")


def select_threshold(y_validation: np.ndarray, probabilities: np.ndarray) -> float:
    """Pick the validation threshold that maximizes fraud-class F1."""
    if len(y_validation) != len(probabilities) or len(y_validation) == 0:
        raise ValueError("Validation labels and probabilities must be non-empty and aligned")
    _require_binary_labels(y_validation, "Validation")
    if np.unique(y_validation).size != 2:
        raise ValueError(
            "Validation labels must contain both classes to select an F1 threshold"
        )
    precision, recall, thresholds = precision_recall_curve(
        y_validation, probabilities, pos_label=FRAUD_LABEL
    )
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int | list[list[int]]]:
    """Score test predictions for the minority fraud class only.

    Precision, recall, F1, and PR-AUC treat ``isFraud == 1`` as the positive
    class; majority-class, macro, and weighted averages are never reported.
    """
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and probabilities must be aligned")
    _require_binary_labels(y_true, "Test")
    if np.unique(y_true).size != 2:
        raise ValueError("Test labels must contain both classes to calculate ROC-AUC")
    predicted = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predicted, labels=[0, FRAUD_LABEL])
    (_, false_positives), (false_negatives, true_positives) = matrix
    return {
        "pr_auc": float(
            average_precision_score(y_true, probabilities, pos_label=FRAUD_LABEL)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, **FRAUD_CLASS_SCORING)),
        "recall": float(recall_score(y_true, predicted, **FRAUD_CLASS_SCORING)),
        "f1": float(f1_score(y_true, predicted, **FRAUD_CLASS_SCORING)),
        "fraud_support": int(true_positives + false_negatives),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_with_validation_threshold(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> tuple[float, dict[str, float | int | list[list[int]]]]:
    threshold = select_threshold(y_validation, validation_probabilities)
    return threshold, evaluate_predictions(y_test, test_probabilities, threshold)
