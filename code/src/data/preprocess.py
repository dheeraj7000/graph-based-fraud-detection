from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
}
TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
IDENTIFIER_COLUMNS = ("nameOrig", "nameDest")
TARGET_COLUMN = "isFraud"


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float32",
            "nameOrig": "category",
            "oldbalanceOrg": "float32",
            "newbalanceOrig": "float32",
            "nameDest": "category",
            "oldbalanceDest": "float32",
            "newbalanceDest": "float32",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        },
    )
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required PaySim columns contain missing values")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("isFraud must contain only binary 0/1 labels")
    return frame


def make_tabular_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build ID-free predictors; target and isFlaggedFraud are never included."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features = pd.DataFrame(index=frame.index)
    for col in (
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ):
        features[col] = pd.to_numeric(frame[col], errors="raise").astype("float32")

    features["origin_balance_change"] = (
        features["newbalanceOrig"] - features["oldbalanceOrg"]
    )
    features["destination_balance_change"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )
    features["amount_to_origin_balance_ratio"] = (
        features["amount"] / features["oldbalanceOrg"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    features["amount_to_destination_balance_ratio"] = (
        features["amount"] / features["oldbalanceDest"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    features["origin_zero_balance_indicator"] = (
        features["oldbalanceOrg"] == 0
    ).astype("int8")
    features["destination_zero_balance_indicator"] = (
        features["oldbalanceDest"] == 0
    ).astype("int8")

    for transaction_type in TRANSACTION_TYPES:
        features[f"type_{transaction_type.lower()}"] = (
            frame["type"].eq(transaction_type).astype("int8")
        )
    unknown_type = ~frame["type"].isin(TRANSACTION_TYPES)
    if unknown_type.any():
        types = sorted(frame.loc[unknown_type, "type"].astype(str).unique())
        raise ValueError(f"Unexpected transaction types: {types}")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Engineered tabular predictors contain non-finite values")
    return features


def read_data(path: str | Path) -> pd.DataFrame:
    """Compatibility entry point used by pipeline scripts."""
    return load_transactions(path)
