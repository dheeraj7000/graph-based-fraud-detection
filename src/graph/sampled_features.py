from __future__ import annotations

import networkx as nx
import pandas as pd

from src.graph.static_features import (
    causal_static_prefix_features,
    training_graph_transactions,
)

SAMPLED_GRAPH_COLUMNS = [
    f"{role}_{metric}"
    for role in ("source", "destination")
    for metric in (
        "pagerank",
        "in_degree",
        "out_degree",
        "weighted_degree",
        "unique_counterparty",
        "community_id",
        "unseen",
    )
]


def sample_training_edges(
    transactions: pd.DataFrame,
    train_end_step: int,
    max_edges: int,
    seed: int,
) -> pd.DataFrame:
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    training = training_graph_transactions(transactions, train_end_step)
    if len(training) <= max_edges:
        return training
    return training.sample(n=max_edges, random_state=seed).sort_index()


def sampled_graph_account_features(
    sampled_training: pd.DataFrame,
) -> pd.DataFrame:
    required = {"nameOrig", "nameDest", "amount"}
    missing = required.difference(sampled_training.columns)
    if missing:
        raise ValueError(f"Sampled graph input missing columns: {sorted(missing)}")
    if sampled_training.empty:
        raise ValueError("Sampled graph requires at least one training edge")

    graph = nx.DiGraph()
    in_degree: dict[str, int] = {}
    out_degree: dict[str, int] = {}
    in_weight: dict[str, float] = {}
    out_weight: dict[str, float] = {}
    neighbors: dict[str, set[str]] = {}
    for source, destination, amount in sampled_training[
        ["nameOrig", "nameDest", "amount"]
    ].itertuples(index=False, name=None):
        source = str(source)
        destination = str(destination)
        amount = float(amount)
        previous = graph.get_edge_data(source, destination, {}).get("weight", 0.0)
        graph.add_edge(source, destination, weight=previous + amount)
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[destination] = in_degree.get(destination, 0) + 1
        out_weight[source] = out_weight.get(source, 0.0) + amount
        in_weight[destination] = in_weight.get(destination, 0.0) + amount
        neighbors.setdefault(source, set()).add(destination)
        neighbors.setdefault(destination, set()).add(source)

    pagerank = nx.pagerank(graph, weight="weight", max_iter=100)
    community_id: dict[str, int] = {}
    for component, accounts in enumerate(
        nx.connected_components(graph.to_undirected())
    ):
        for account in accounts:
            community_id[account] = component

    accounts = sorted(graph.nodes)
    return pd.DataFrame(
        {
            "account_id": accounts,
            "pagerank": [pagerank.get(account, 0.0) for account in accounts],
            "in_degree": [in_degree.get(account, 0) for account in accounts],
            "out_degree": [out_degree.get(account, 0) for account in accounts],
            "weighted_degree": [
                in_weight.get(account, 0.0) + out_weight.get(account, 0.0)
                for account in accounts
            ],
            "unique_counterparty": [
                len(neighbors.get(account, ())) for account in accounts
            ],
            "community_id": [community_id.get(account, -1) for account in accounts],
        }
    )


def attach_sampled_graph_features(
    transactions: pd.DataFrame,
    sampled_training: pd.DataFrame,
    account_features: pd.DataFrame,
    train_end_step: int,
) -> pd.DataFrame:
    if not sampled_training.index.isin(transactions.index).all():
        raise ValueError("Sampled training row indexes must belong to transactions")
    features = pd.DataFrame(0.0, index=transactions.index, columns=SAMPLED_GRAPH_COLUMNS)
    for role in ("source", "destination"):
        features[f"{role}_community_id"] = -1
        features[f"{role}_unseen"] = 1

    indexed = account_features.set_index("account_id")
    validation_test = transactions["step"] > train_end_step
    for role, identifier in (("source", "nameOrig"), ("destination", "nameDest")):
        ids = transactions.loc[validation_test, identifier].astype(str)
        for metric in (
            "pagerank",
            "in_degree",
            "out_degree",
            "weighted_degree",
            "unique_counterparty",
            "community_id",
        ):
            default = -1 if metric == "community_id" else 0
            features.loc[validation_test, f"{role}_{metric}"] = (
                ids.map(indexed[metric]).fillna(default).to_numpy()
            )
        features.loc[validation_test, f"{role}_unseen"] = (
            ~ids.isin(indexed.index)
        ).astype("int8").to_numpy()

    causal = causal_static_prefix_features(sampled_training, train_end_step)
    for role in ("source", "destination"):
        for metric in (
            "in_degree",
            "out_degree",
            "weighted_degree",
            "unique_counterparty",
            "unseen",
        ):
            features.loc[sampled_training.index, f"{role}_{metric}"] = causal[
                f"{role}_{metric}"
            ].to_numpy()
    return features
