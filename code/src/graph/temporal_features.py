from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb


@dataclass
class _WindowTotals:
    count: int = 0
    value: float = 0.0
    counterpart_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counterpart_counts is None:
            self.counterpart_counts = {}


class RollingTemporalFeatures:
    def __init__(self, window: int):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.totals: dict[tuple[str, str], _WindowTotals] = {}
        self.events: deque[
            tuple[int, list[tuple[str, str, int, float, dict[str, int]]]]
        ] = deque()
        self.last_seen: dict[str, int] = {}

    def process_step(self, current: pd.DataFrame) -> pd.DataFrame:
        if current.empty:
            return pd.DataFrame()
        if current["step"].nunique() != 1:
            raise ValueError("process_step requires transactions from exactly one step")
        step = int(current["step"].iloc[0])
        while self.events and self.events[0][0] < step - self.window:
            _, expired = self.events.popleft()
            for account, direction, count, value, counterparties in expired:
                state = self.totals[(account, direction)]
                state.count -= count
                state.value -= value
                assert state.counterpart_counts is not None
                for counterparty, occurrences in counterparties.items():
                    remaining = state.counterpart_counts[counterparty] - occurrences
                    if remaining:
                        state.counterpart_counts[counterparty] = remaining
                    else:
                        del state.counterpart_counts[counterparty]

        source_ids = current["nameOrig"].astype(str)
        destination_ids = current["nameDest"].astype(str)
        source_accounts = source_ids.to_numpy()
        destination_accounts = destination_ids.to_numpy()
        row_ids = current["_row_id"].to_numpy()

        incoming_states = [self.totals.get((account, "in")) for account in destination_ids]
        outgoing_states = [self.totals.get((account, "out")) for account in source_ids]
        last_source = [self.last_seen.get(account, -1) for account in source_ids]
        last_destination = [
            self.last_seen.get(account, -1) for account in destination_ids
        ]

        output = pd.DataFrame(
            {
                "_row_id": row_ids,
                f"incoming_tx_count_{self.window}h": np.fromiter(
                    (state.count if state else 0 for state in incoming_states),
                    dtype=np.int32,
                    count=len(current),
                ),
                f"incoming_value_{self.window}h": np.fromiter(
                    (state.value if state else 0.0 for state in incoming_states),
                    dtype=np.float32,
                    count=len(current),
                ),
                f"unique_senders_{self.window}h": np.fromiter(
                    (
                        len(state.counterpart_counts or {}) if state else 0
                        for state in incoming_states
                    ),
                    dtype=np.int32,
                    count=len(current),
                ),
                f"outgoing_tx_count_{self.window}h": np.fromiter(
                    (state.count if state else 0 for state in outgoing_states),
                    dtype=np.int32,
                    count=len(current),
                ),
                f"outgoing_value_{self.window}h": np.fromiter(
                    (state.value if state else 0.0 for state in outgoing_states),
                    dtype=np.float32,
                    count=len(current),
                ),
                f"unique_receivers_{self.window}h": np.fromiter(
                    (
                        len(state.counterpart_counts or {}) if state else 0
                        for state in outgoing_states
                    ),
                    dtype=np.int32,
                    count=len(current),
                ),
                "time_since_origin_transaction": np.fromiter(
                    (step - value if value >= 0 else -1 for value in last_source),
                    dtype=np.int32,
                    count=len(current),
                ),
                "time_since_destination_transaction": np.fromiter(
                    (
                        step - value if value >= 0 else -1
                        for value in last_destination
                    ),
                    dtype=np.int32,
                    count=len(current),
                ),
            }
        )

        by_direction: dict[
            tuple[str, str], tuple[int, float, dict[str, int]]
        ] = {}
        for direction, accounts, counterparties in (
            ("out", source_accounts, destination_accounts),
            ("in", destination_accounts, source_accounts),
        ):
            pairs = pd.DataFrame(
                {
                    "account": accounts,
                    "counterparty": counterparties,
                    "amount": current["amount"].to_numpy(dtype=np.float32),
                }
            )
            grouped = pairs.groupby(
                ["account", "counterparty"], sort=False, observed=True
            )["amount"].agg(["size", "sum"])
            account_events = grouped.groupby(level="account", sort=False).agg(
                count=("size", "sum"), value=("sum", "sum")
            )
            counterpart_maps = grouped.groupby(level="account", sort=False)[
                "size"
            ].agg(
                lambda account_pairs: dict(
                    zip(
                        account_pairs.index.get_level_values("counterparty"),
                        account_pairs.to_numpy(),
                    )
                )
            )
            for account, count, value, counterparty_counts in zip(
                account_events.index,
                account_events["count"].to_numpy(),
                account_events["value"].to_numpy(),
                counterpart_maps.to_numpy(),
            ):
                by_direction[(str(account), direction)] = (
                    int(count),
                    float(value),
                    {
                        str(counterparty): int(count)
                        for counterparty, count in counterparty_counts.items()
                    },
                )

        expiry_events: list[tuple[str, str, int, float, dict[str, int]]] = []
        for (account, direction), (count, value, counterparties) in by_direction.items():
            state = self.totals.setdefault((account, direction), _WindowTotals())
            state.count += count
            state.value += value
            assert state.counterpart_counts is not None
            for counterparty, occurrences in counterparties.items():
                state.counterpart_counts[counterparty] = (
                    state.counterpart_counts.get(counterparty, 0) + occurrences
                )
            expiry_events.append((account, direction, count, value, counterparties))
        self.events.append((step, expiry_events))
        for account in set(source_accounts).union(destination_accounts):
            self.last_seen[account] = step
        return output


def build_temporal_features(
    transactions: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    """Compute [step-window, step) features with same-step updates batched."""
    required = {"step", "amount", "nameOrig", "nameDest"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Temporal feature input missing columns: {sorted(missing)}")
    ordered = transactions.loc[:, ["step", "amount", "nameOrig", "nameDest"]].copy()
    ordered["_row_id"] = range(len(ordered))
    ordered = ordered.sort_values(["step", "_row_id"], kind="stable")
    state = RollingTemporalFeatures(window)
    records = [
        state.process_step(current)
        for _, current in ordered.groupby("step", sort=True)
    ]
    result = pd.concat(records, ignore_index=True).sort_values("_row_id")
    return result.drop(columns="_row_id").reset_index(drop=True)


def stream_temporal_features(
    input_path: str | Path,
    output_path: str | Path,
    window: int = 24,
) -> int:
    """Generate causal rolling features with bounded-memory DuckDB window queries."""
    if window <= 0:
        raise ValueError("window must be positive")
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    temporary_dir = Path("data/processed/duckdb_tmp")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    input_sql_path = str(input_path.resolve()).replace("\\", "/").replace("'", "''")
    output_sql_path = str(output_path.resolve()).replace("\\", "/").replace("'", "''")
    temp_sql_path = str(temporary_dir.resolve()).replace("\\", "/").replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET memory_limit = '2GB'")
        connection.execute("SET threads = 4")
        connection.execute(f"SET temp_directory = '{temp_sql_path}'")
        connection.execute(
            f"""
            CREATE TABLE transactions AS
            SELECT
                CAST(ordinality - 1 AS BIGINT) AS _row_id,
                CAST(step AS INTEGER) AS step,
                CAST(amount AS DOUBLE) AS amount,
                CAST(nameOrig AS VARCHAR) AS nameOrig,
                CAST(nameDest AS VARCHAR) AS nameDest
            FROM read_csv(
                '{input_sql_path}',
                header = true,
                auto_detect = true,
                sample_size = -1
            ) WITH ORDINALITY
            """
        )
        row_count = int(
            connection.execute("SELECT count(*) FROM transactions").fetchone()[0]
        )
        if row_count == 0:
            raise ValueError("PaySim CSV contains no transaction rows")
        disorder_count = int(
            connection.execute(
                """
                SELECT count(*)
                FROM (
                    SELECT step, lag(step) OVER (ORDER BY _row_id) AS previous_step
                    FROM transactions
                )
                WHERE step < previous_step
                """
            ).fetchone()[0]
        )
        if disorder_count:
            raise ValueError("PaySim CSV rows must be chronologically ordered")

        interval = int(window)
        query = f"""
            COPY (
                WITH incoming AS (
                    SELECT
                        _row_id,
                        step,
                        CAST(count(*) OVER history_window AS INTEGER)
                            AS incoming_tx_count_{interval}h,
                        CAST(sum(amount) OVER history_window AS FLOAT)
                            AS incoming_value_{interval}h,
                        CAST(count(DISTINCT nameOrig) OVER history_window AS INTEGER)
                            AS unique_senders_{interval}h,
                        CAST(max(step) OVER all_history AS INTEGER)
                            AS previous_destination_step
                    FROM transactions
                    WINDOW
                        history_window AS (
                            PARTITION BY nameDest
                            ORDER BY step
                            RANGE BETWEEN {interval} PRECEDING AND 1 PRECEDING
                        ),
                        all_history AS (
                            PARTITION BY nameDest
                            ORDER BY step
                            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        )
                ),
                outgoing AS (
                    SELECT
                        _row_id,
                        CAST(count(*) OVER history_window AS INTEGER)
                            AS outgoing_tx_count_{interval}h,
                        CAST(sum(amount) OVER history_window AS FLOAT)
                            AS outgoing_value_{interval}h,
                        CAST(count(DISTINCT nameDest) OVER history_window AS INTEGER)
                            AS unique_receivers_{interval}h,
                        CAST(max(step) OVER all_history AS INTEGER)
                            AS previous_origin_step
                    FROM transactions
                    WINDOW
                        history_window AS (
                            PARTITION BY nameOrig
                            ORDER BY step
                            RANGE BETWEEN {interval} PRECEDING AND 1 PRECEDING
                        ),
                        all_history AS (
                            PARTITION BY nameOrig
                            ORDER BY step
                            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        )
                )
                SELECT
                    incoming._row_id,
                    incoming.incoming_tx_count_{interval}h,
                    coalesce(incoming.incoming_value_{interval}h, 0.0)
                        AS incoming_value_{interval}h,
                    incoming.unique_senders_{interval}h,
                    outgoing.outgoing_tx_count_{interval}h,
                    coalesce(outgoing.outgoing_value_{interval}h, 0.0)
                        AS outgoing_value_{interval}h,
                    outgoing.unique_receivers_{interval}h,
                    coalesce(incoming.step - outgoing.previous_origin_step, -1)
                        AS time_since_origin_transaction,
                    coalesce(incoming.step - incoming.previous_destination_step, -1)
                        AS time_since_destination_transaction
                FROM incoming
                JOIN outgoing USING (_row_id)
                ORDER BY incoming._row_id
            )
            TO '{output_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        connection.execute(query)
        written_count = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{output_sql_path}')"
            ).fetchone()[0]
        )
        if written_count != row_count:
            raise AssertionError(
                f"Temporal output row count mismatch: {written_count} != {row_count}"
            )
        return written_count
    finally:
        connection.close()


def max_history_step_for_transaction(
    history_steps: list[int], step: int
) -> int | None:
    prior = [history_step for history_step in history_steps if history_step < step]
    return max(prior) if prior else None
