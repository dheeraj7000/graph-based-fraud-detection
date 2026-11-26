from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from src.graph.static_features import (
    causal_static_prefix_features,
    training_graph_transactions,
)

GRAPH_KINDS = ("sampled", "full")
ACCOUNT_METRICS = (
    "pagerank",
    "in_degree",
    "out_degree",
    "weighted_degree",
    "unique_counterparty",
    "community_id",
)
CAUSAL_METRICS = (
    "in_degree",
    "out_degree",
    "weighted_degree",
    "unique_counterparty",
    "unseen",
)
GRAPH_COLUMNS = [
    f"{role}_{metric}"
    for role in ("source", "destination")
    for metric in (*ACCOUNT_METRICS, "unseen")
]


def select_training_edges(
    transactions: pd.DataFrame,
    train_end_step: int,
    max_edges: int | None,
    seed: int,
) -> pd.DataFrame:
    """Return training-period edges, optionally capped by a seeded random sample."""
    if max_edges is not None and max_edges < 1:
        raise ValueError("max_edges must be positive or None for the full graph")
    training = training_graph_transactions(transactions, train_end_step)
    if max_edges is None or len(training) <= max_edges:
        return training
    return training.sample(n=max_edges, random_state=seed).sort_index()


def weighted_pagerank(
    sources: np.ndarray,
    destinations: np.ndarray,
    weights: np.ndarray,
    node_count: int,
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """Sparse power-iteration PageRank with NetworkX's weighted semantics.

    Parallel edges are summed, nodes without positive out-weight redistribute
    their rank uniformly, and teleportation is uniform. ``tol`` is the total L1
    change between iterations, independent of graph size.
    """
    if node_count < 1:
        raise ValueError("PageRank requires at least one node")
    adjacency = sparse.coo_array(
        (np.asarray(weights, dtype=np.float64), (sources, destinations)),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency.sum_duplicates()
    rows = np.repeat(np.arange(node_count), np.diff(adjacency.indptr))
    out_weight = np.bincount(rows, weights=adjacency.data, minlength=node_count)
    dangling = out_weight <= 0
    scaled = np.divide(
        adjacency.data,
        out_weight[rows],
        out=np.zeros_like(adjacency.data),
        where=out_weight[rows] > 0,
    )
    incoming = sparse.coo_array(
        (scaled, (adjacency.indices, rows)), shape=(node_count, node_count)
    ).tocsr()
    rank = np.full(node_count, 1.0 / node_count)
    for _ in range(max_iter):
        previous = rank
        rank = alpha * (incoming @ previous + previous[dangling].sum() / node_count)
        rank += (1.0 - alpha) / node_count
        if np.abs(rank - previous).sum() < tol:
            return rank
    raise RuntimeError(f"PageRank did not converge within {max_iter} iterations")


def account_graph_features(training_edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize each account in the directed, amount-weighted training graph."""
    required = {"nameOrig", "nameDest", "amount"}
    missing = required.difference(training_edges.columns)
    if missing:
        raise ValueError(f"Graph input missing columns: {sorted(missing)}")
    if training_edges.empty:
        raise ValueError("Account graph requires at least one training edge")

    edge_count = len(training_edges)
    codes, accounts = pd.factorize(
        np.concatenate(
            [
                training_edges["nameOrig"].astype(str).to_numpy(dtype=object),
                training_edges["nameDest"].astype(str).to_numpy(dtype=object),
            ]
        ),
        sort=True,
    )
    node_count = len(accounts)
    sources = codes[:edge_count].astype(np.int64)
    destinations = codes[edge_count:].astype(np.int64)
    amounts = training_edges["amount"].to_numpy(dtype=np.float64)

    out_degree = np.bincount(sources, minlength=node_count)
    in_degree = np.bincount(destinations, minlength=node_count)
    weighted_degree = np.bincount(
        sources, weights=amounts, minlength=node_count
    ) + np.bincount(destinations, weights=amounts, minlength=node_count)

    low = np.minimum(sources, destinations)
    high = np.maximum(sources, destinations)
    pairs = np.unique(low * node_count + high)
    pair_low, pair_high = np.divmod(pairs, node_count)
    unique_counterparty = np.bincount(pair_low, minlength=node_count) + np.bincount(
        pair_high[pair_high != pair_low], minlength=node_count
    )

    structure = sparse.coo_array(
        (np.ones(edge_count, dtype=np.int8), (sources, destinations)),
        shape=(node_count, node_count),
    ).tocsr()
    _, community_id = connected_components(
        structure, directed=True, connection="weak"
    )
    pagerank = weighted_pagerank(sources, destinations, amounts, node_count)

    return pd.DataFrame(
        {
            "account_id": np.asarray(accounts, dtype=object),
            "pagerank": pagerank,
            "in_degree": in_degree.astype(np.int64),
            "out_degree": out_degree.astype(np.int64),
            "weighted_degree": weighted_degree,
            "unique_counterparty": unique_counterparty.astype(np.int64),
            "community_id": community_id.astype(np.int64),
        }
    )


def attach_graph_features(
    transactions: pd.DataFrame,
    training_edges: pd.DataFrame,
    account_features: pd.DataFrame,
    train_end_step: int,
) -> pd.DataFrame:
    """Join account features to each transaction without exposing future edges.

    Validation/test rows read the frozen training graph. Graph training rows
    get causal-prefix aggregates only; their global PageRank and component IDs
    stay at defaults because those summaries include later training edges.
    """
    if not training_edges.index.isin(transactions.index).all():
        raise ValueError("Training edge row indexes must belong to transactions")
    features = pd.DataFrame(0.0, index=transactions.index, columns=GRAPH_COLUMNS)
    for role in ("source", "destination"):
        features[f"{role}_community_id"] = -1
        features[f"{role}_unseen"] = 1

    account_index = pd.Index(account_features["account_id"].astype(str))
    held_out = (transactions["step"] > train_end_step).to_numpy()
    for role, identifier in (("source", "nameOrig"), ("destination", "nameDest")):
        positions = account_index.get_indexer(
            transactions.loc[held_out, identifier].astype(str)
        )
        known = positions >= 0
        for metric in ACCOUNT_METRICS:
            default = -1 if metric == "community_id" else 0
            values = account_features[metric].to_numpy()
            features.loc[held_out, f"{role}_{metric}"] = np.where(
                known, values[positions], default
            )
        features.loc[held_out, f"{role}_unseen"] = (~known).astype(np.int8)

    causal = causal_static_prefix_features(training_edges, train_end_step)
    for role in ("source", "destination"):
        for metric in CAUSAL_METRICS:
            column = f"{role}_{metric}"
            features.loc[training_edges.index, column] = causal[column].to_numpy()
    return features


def build_graph_feature_cache(
    transactions: pd.DataFrame,
    train_end_step: int,
    kind: str,
    seed: int,
    edge_limit: int | None = None,
    feature_dir: str | Path = "data/features",
    processed_dir: str | Path = "data/processed",
) -> dict[str, Any]:
    """Write ``{kind}_graph_*`` caches; ``full`` uses every training edge."""
    if kind not in GRAPH_KINDS:
        raise ValueError(f"Unknown graph kind {kind!r}; expected one of {GRAPH_KINDS}")
    if kind == "full" and edge_limit is not None:
        raise ValueError("The full graph uses every training edge; omit edge_limit")
    if kind == "sampled" and edge_limit is None:
        raise ValueError("The sampled graph requires an edge_limit")
    feature_dir = Path(feature_dir)
    processed_dir = Path(processed_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    training_edges = select_training_edges(
        transactions, train_end_step=train_end_step, max_edges=edge_limit, seed=seed
    )
    account_features = account_graph_features(training_edges)
    row_features = attach_graph_features(
        transactions, training_edges, account_features, train_end_step
    )
    row_features.insert(0, "_row_id", transactions.index.astype("int64"))
    row_features.to_parquet(feature_dir / f"{kind}_graph_features.parquet", index=False)
    account_features.to_parquet(
        feature_dir / f"{kind}_graph_account_features.parquet", index=False
    )
    if kind == "sampled":
        pd.DataFrame({"_row_id": training_edges.index.astype("int64")}).to_parquet(
            processed_dir / "sampled_graph_train_rows.parquet", index=False
        )
    metadata = {
        "kind": kind,
        "backend": "SciPy sparse PageRank/components + DuckDB causal prefix",
        "seed": seed,
        "edge_limit": edge_limit,
        f"{kind}_training_edges": len(training_edges),
        f"{kind}_training_frauds": int(training_edges["isFraud"].sum()),
        f"{kind}_training_accounts": len(account_features),
        f"{kind}_weak_components": int(account_features["community_id"].nunique()),
        "train_end_step": int(train_end_step),
    }
    (processed_dir / f"{kind}_graph_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
