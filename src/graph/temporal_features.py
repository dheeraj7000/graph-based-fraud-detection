from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class _WindowTotals:
    count: int = 0
    value: float = 0.0
    counterpart_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counterpart_counts is None:
            self.counterpart_counts = {}


def build_temporal_features(
    transactions: pd.DataFrame, window: int = 24
) -> pd.DataFrame:
    """Compute prior-window account activity; same-step transactions are batched."""
    required = {"step", "amount", "nameOrig", "nameDest"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Temporal feature input missing columns: {sorted(missing)}")
    if window <= 0:
        raise ValueError("window must be positive")

    ordered = transactions.loc[:, ["step", "amount", "nameOrig", "nameDest"]].copy()
    ordered["_row_id"] = range(len(ordered))
    ordered = ordered.sort_values(["step", "_row_id"], kind="stable")

    totals: dict[tuple[str, str], _WindowTotals] = {}
    events: deque[tuple[int, list[tuple[str, str, int, float, dict[str, int]]]]] = deque()
    last_seen: dict[str, int] = {}
    records: list[pd.DataFrame] = []

    for step_value, current in ordered.groupby("step", sort=True):
        step = int(step_value)
        while events and events[0][0] < step - window:
            _, expired = events.popleft()
            for account, direction, count, value, counterparties in expired:
                state = totals[(account, direction)]
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
        source_state = pd.DataFrame({"_row_id": current["_row_id"], "account": source_ids})
        destination_state = pd.DataFrame(
            {"_row_id": current["_row_id"], "account": destination_ids}
        )

        for direction, data in (("in", destination_state), ("out", source_state)):
            states = [totals.get((account, direction)) for account in data["account"]]
            data["count"] = [state.count if state else 0 for state in states]
            data["value"] = [state.value if state else 0.0 for state in states]
            data["unique"] = [
                len(state.counterpart_counts or {}) if state else 0 for state in states
            ]
            if direction == "in":
                data = data.rename(
                    columns={
                        "count": f"incoming_tx_count_{window}h",
                        "value": f"incoming_value_{window}h",
                        "unique": f"unique_senders_{window}h",
                    }
                )
                incoming = data
            else:
                data = data.rename(
                    columns={
                        "count": f"outgoing_tx_count_{window}h",
                        "value": f"outgoing_value_{window}h",
                        "unique": f"unique_receivers_{window}h",
                    }
                )
                outgoing = data

        last_source = source_ids.map(last_seen).fillna(-1).astype("int64")
        last_destination = destination_ids.map(last_seen).fillna(-1).astype("int64")
        base = pd.DataFrame(
            {
                "_row_id": current["_row_id"].to_numpy(),
                "step": step,
                "time_since_origin_transaction": [
                    step - previous if previous >= 0 else -1
                    for previous in last_source
                ],
                "time_since_destination_transaction": [
                    step - previous if previous >= 0 else -1
                    for previous in last_destination
                ],
            }
        )
        step_features = (
            base.merge(incoming.drop(columns="account"), on="_row_id", validate="one_to_one")
            .merge(outgoing.drop(columns="account"), on="_row_id", validate="one_to_one")
        )
        records.append(step_features)

        current_events: dict[
            tuple[str, str], tuple[int, float, dict[str, int]]
        ] = {}
        origin_events = pd.DataFrame(
            {
                "account": source_ids.to_numpy(),
                "counterparty": destination_ids.to_numpy(),
                "amount": current["amount"].to_numpy(),
            }
        )
        destination_events = pd.DataFrame(
            {
                "account": destination_ids.to_numpy(),
                "counterparty": source_ids.to_numpy(),
                "amount": current["amount"].to_numpy(),
            }
        )
        for direction, data in (("out", origin_events), ("in", destination_events)):
            grouped = data.groupby(["account", "counterparty"], sort=False).agg(
                count=("amount", "size"), value=("amount", "sum")
            )
            account_events = grouped.groupby(level="account", sort=False).agg(
                count=("count", "sum"), value=("value", "sum")
            )
            counterpart_maps = grouped["count"].groupby(level="account", sort=False).apply(
                lambda values: values.droplevel("account").to_dict()
            )
            for account, row in account_events.iterrows():
                current_events[(str(account), direction)] = (
                    int(row["count"]),
                    float(row["value"]),
                    {str(k): int(v) for k, v in counterpart_maps.loc[account].items()},
                )

        expiry_events: list[tuple[str, str, int, float, dict[str, int]]] = []
        for (account, direction), (count, value, counterparties) in current_events.items():
            state = totals.setdefault((account, direction), _WindowTotals())
            state.count += count
            state.value += value
            assert state.counterpart_counts is not None
            for counterparty, occurrences in counterparties.items():
                state.counterpart_counts[counterparty] = (
                    state.counterpart_counts.get(counterparty, 0) + occurrences
                )
            expiry_events.append((account, direction, count, value, counterparties))
        events.append((step, expiry_events))

        touched_accounts = pd.concat([source_ids, destination_ids]).unique()
        for account in touched_accounts:
            last_seen[str(account)] = step

    result = pd.concat(records, ignore_index=True).sort_values("_row_id")
    result = result.drop(columns=["_row_id", "step"]).reset_index(drop=True)
    return result


def max_history_step_for_transaction(history_steps: list[int], step: int) -> int | None:
    """Small invariant helper used by leakage tests."""
    prior = [history_step for history_step in history_steps if history_step < step]
    return max(prior) if prior else None
