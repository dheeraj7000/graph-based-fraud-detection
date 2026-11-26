from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def training_graph_transactions(
    transactions: pd.DataFrame, train_end_step: int
) -> pd.DataFrame:
    training = transactions.loc[transactions["step"] <= train_end_step].copy()
    if not training.empty and int(training["step"].max()) > train_end_step:
        raise AssertionError("A graph edge exceeds the configured training cutoff")
    return training


def causal_unique_counterparty_counts(
    transactions: pd.DataFrame, train_end_step: int
) -> tuple[np.ndarray, np.ndarray]:
    prefix = causal_static_prefix_features(transactions, train_end_step)
    return (
        prefix["source_unique_counterparty"].to_numpy(),
        prefix["destination_unique_counterparty"].to_numpy(),
    )


PREFIX_METRICS = (
    "in_degree",
    "out_degree",
    "weighted_degree",
    "unique_counterparty",
    "unseen",
)


def _prefix_dtype(metric: str) -> type[np.generic]:
    if metric == "weighted_degree":
        return np.float64
    if metric == "unseen":
        return np.int8
    return np.int32


def causal_static_prefix_features(
    transactions: pd.DataFrame, train_end_step: int
) -> pd.DataFrame:
    """Build prior-only account aggregates, updating history only after each step.

    Each account's history covers steps strictly before the current step, in
    both source and destination roles. Rows after ``train_end_step`` stay zero.
    """
    required = {"step", "amount", "nameOrig", "nameDest"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Static feature input missing columns: {sorted(missing)}")
    size = len(transactions)
    stats = {
        f"{role}_{metric}": np.zeros(size, dtype=_prefix_dtype(metric))
        for role in ("source", "destination")
        for metric in PREFIX_METRICS
    }
    if size == 0:
        return pd.DataFrame(stats, index=transactions.index)

    account_codes, _ = pd.factorize(
        np.concatenate(
            [
                transactions["nameOrig"].astype(str).to_numpy(dtype=object),
                transactions["nameDest"].astype(str).to_numpy(dtype=object),
            ]
        )
    )
    edges = pd.DataFrame(
        {
            "_pos": np.arange(size, dtype=np.int64),
            "step": transactions["step"].to_numpy(dtype=np.int64),
            "amount": transactions["amount"].to_numpy(dtype=np.float64),
            "source": account_codes[:size].astype(np.int64),
            "destination": account_codes[size:].astype(np.int64),
        }
    )
    temporary_dir = Path("data/processed/duckdb_tmp")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temp_sql_path = str(temporary_dir.resolve()).replace("\\", "/").replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET memory_limit = '4GB'")
        connection.execute(f"SET temp_directory = '{temp_sql_path}'")
        connection.register("edges", edges)
        history = connection.execute(
            f"""
            WITH events AS (
                SELECT _pos, 0 AS role, source AS account, destination AS counterparty,
                       step, amount, 0 AS is_in, 1 AS is_out
                FROM edges WHERE step <= {int(train_end_step)}
                UNION ALL
                SELECT _pos, 1 AS role, destination AS account, source AS counterparty,
                       step, amount, 1 AS is_in, 0 AS is_out
                FROM edges WHERE step <= {int(train_end_step)}
            )
            SELECT
                _pos,
                role,
                CAST(coalesce(sum(is_in) OVER prior, 0) AS INTEGER) AS in_degree,
                CAST(coalesce(sum(is_out) OVER prior, 0) AS INTEGER) AS out_degree,
                coalesce(sum(amount) OVER prior, 0.0) AS weighted_degree,
                CAST(count(DISTINCT counterparty) OVER prior AS INTEGER)
                    AS unique_counterparty,
                CAST((count(*) OVER prior) = 0 AS TINYINT) AS unseen
            FROM events
            WINDOW prior AS (
                PARTITION BY account
                ORDER BY step
                RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
            """
        ).df()
    finally:
        connection.close()

    for role_code, role in ((0, "source"), (1, "destination")):
        current = history.loc[history["role"] == role_code]
        positions = current["_pos"].to_numpy()
        for metric in PREFIX_METRICS:
            stats[f"{role}_{metric}"][positions] = current[metric].to_numpy(
                dtype=_prefix_dtype(metric)
            )
    return pd.DataFrame(stats, index=transactions.index)
