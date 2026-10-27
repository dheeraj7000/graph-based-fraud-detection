from __future__ import annotations

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


def causal_static_prefix_features(
    transactions: pd.DataFrame, train_end_step: int
) -> pd.DataFrame:
    """Build prior-only account aggregates, updating history only after each step."""
    required = {"step", "amount", "nameOrig", "nameDest"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"Static feature input missing columns: {sorted(missing)}")
    ordered = transactions.loc[:, ["step", "amount", "nameOrig", "nameDest"]].copy()
    ordered["_row_id"] = range(len(ordered))
    ordered = ordered.sort_values(["step", "_row_id"], kind="stable")
    size = len(ordered)
    stats = {
        f"{role}_{name}": np.zeros(
            size,
            dtype=(
                np.float64
                if name == "weighted_degree"
                else np.int8
                if name == "unseen"
                else np.int32
            ),
        )
        for role in ("source", "destination")
        for name in (
            "in_degree",
            "out_degree",
            "weighted_degree",
            "unique_counterparty",
            "unseen",
        )
    }
    in_degree: dict[str, int] = {}
    out_degree: dict[str, int] = {}
    in_value: dict[str, float] = {}
    out_value: dict[str, float] = {}
    neighbors: dict[str, set[str]] = {}
    seen_accounts: set[str] = set()

    for step, current in ordered.groupby("step", sort=True):
        if int(step) > train_end_step:
            break
        source_ids = current["nameOrig"].astype(str)
        destination_ids = current["nameDest"].astype(str)
        row_ids = current["_row_id"].to_numpy()
        amounts = current["amount"].to_numpy(dtype=np.float64)
        for role, accounts in (
            ("source", source_ids),
            ("destination", destination_ids),
        ):
            stats[f"{role}_in_degree"][row_ids] = [
                in_degree.get(account, 0) for account in accounts
            ]
            stats[f"{role}_out_degree"][row_ids] = [
                out_degree.get(account, 0) for account in accounts
            ]
            stats[f"{role}_weighted_degree"][row_ids] = [
                in_value.get(account, 0.0) + out_value.get(account, 0.0)
                for account in accounts
            ]
            stats[f"{role}_unique_counterparty"][row_ids] = [
                len(neighbors.get(account, ())) for account in accounts
            ]
            stats[f"{role}_unseen"][row_ids] = [
                int(account not in seen_accounts) for account in accounts
            ]

        source_events = pd.DataFrame(
            {"account": source_ids.to_numpy(), "amount": amounts}
        ).groupby("account", sort=False)["amount"].agg(["size", "sum"])
        destination_events = pd.DataFrame(
            {"account": destination_ids.to_numpy(), "amount": amounts}
        ).groupby("account", sort=False)["amount"].agg(["size", "sum"])
        for account, event in source_events.iterrows():
            account = str(account)
            out_degree[account] = out_degree.get(account, 0) + int(event["size"])
            out_value[account] = out_value.get(account, 0.0) + float(event["sum"])
        for account, event in destination_events.iterrows():
            account = str(account)
            in_degree[account] = in_degree.get(account, 0) + int(event["size"])
            in_value[account] = in_value.get(account, 0.0) + float(event["sum"])

        current_pairs = pd.DataFrame(
            {"source": source_ids.to_numpy(), "destination": destination_ids.to_numpy()}
        ).drop_duplicates()
        for source, destination in current_pairs.itertuples(index=False, name=None):
            neighbors.setdefault(source, set()).add(destination)
            neighbors.setdefault(destination, set()).add(source)
        seen_accounts.update(source_ids)
        seen_accounts.update(destination_ids)
    return pd.DataFrame(stats, index=transactions.index)
