from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from neo4j import Driver

GRAPH_NAME = "paysimFraudDetectionTrainGraph"


def _batches(frame: pd.DataFrame, size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size].to_dict(orient="records")


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


def build_training_graph(
    driver: Driver,
    transactions: pd.DataFrame,
    train_end_step: int,
    database: str = "neo4j",
    batch_size: int = 10_000,
    pagerank_iterations: int = 20,
) -> pd.DataFrame:
    """Load the train-cutoff graph into Neo4j and return account-level GDS features."""
    training = training_graph_transactions(transactions, train_end_step)
    if training.empty:
        raise ValueError("No transactions fall within the configured graph cutoff")
    training["transaction_id"] = training.index.astype("int64")
    rows = training[
        [
            "transaction_id",
            "nameOrig",
            "nameDest",
            "amount",
            "type",
            "step",
            "isFraud",
        ]
    ].rename(
        columns={
            "nameOrig": "source",
            "nameDest": "destination",
            "isFraud": "fraud",
        }
    )

    with driver.session(database=database) as session:
        session.run(
            f"CALL gds.graph.drop('{GRAPH_NAME}', false) YIELD graphName"
        ).consume()
        while True:
            result = session.run(
                """
                MATCH ()-[r:TRANSACTION {project_id: 'paysim_fraud_detection'}]->()
                WITH r LIMIT $batch DELETE r
                """,
                batch=batch_size,
            )
            deleted = result.consume().counters.relationships_deleted
            if deleted < batch_size:
                break
        while True:
            result = session.run(
                """
                MATCH (n:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
                WHERE NOT (n)--()
                WITH n LIMIT $batch DELETE n
                """,
                batch=batch_size,
            )
            deleted = result.consume().counters.nodes_deleted
            if deleted < batch_size:
                break
        session.run(
            "CREATE CONSTRAINT paysim_account_key IF NOT EXISTS "
            "FOR (a:PaySimFraudAccount) "
            "REQUIRE (a.project_id, a.account_id) IS UNIQUE"
        ).consume()
        for batch in _batches(rows, batch_size):
            session.run(
                """
                UNWIND $rows AS row
                MERGE (source:Account:PaySimFraudAccount {
                    project_id: 'paysim_fraud_detection',
                    account_id: row.source
                })
                MERGE (destination:Account:PaySimFraudAccount {
                    project_id: 'paysim_fraud_detection',
                    account_id: row.destination
                })
                CREATE (source)-[:TRANSACTION {
                    transaction_id: row.transaction_id,
                    project_id: 'paysim_fraud_detection',
                    amount: row.amount,
                    type: row.type,
                    step: row.step,
                    isFraud: row.fraud
                }]->(destination)
                """,
                rows=batch,
            ).consume()

        session.run(
            f"""
            CALL gds.graph.project(
                '{GRAPH_NAME}',
                'PaySimFraudAccount',
                {{TRANSACTION: {{orientation: 'NATURAL', properties: 'amount'}}}}
            )
            """
        ).consume()

        def node_metric(query: str) -> dict[str, Any]:
            return {
                str(record["account_id"]): record["value"]
                for record in session.run(query)
            }

        pagerank = node_metric(
            f"""
            CALL gds.pageRank.stream('{GRAPH_NAME}', {{
                maxIterations: $iterations,
                relationshipWeightProperty: 'amount'
            }})
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).account_id AS account_id, score AS value
            """.replace("$iterations", str(int(pagerank_iterations)))
        )
        communities = node_metric(
            f"""
            CALL gds.louvain.stream('{GRAPH_NAME}', {{
                relationshipWeightProperty: 'amount'
            }})
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).account_id AS account_id,
                   communityId AS value
            """
        )
        in_degree = node_metric(
            """
            MATCH (a:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            OPTIONAL MATCH ()-[r:TRANSACTION {project_id: 'paysim_fraud_detection'}]->(a)
            RETURN a.account_id AS account_id, count(r) AS value
            """
        )
        out_degree = node_metric(
            """
            MATCH (a:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            OPTIONAL MATCH (a)-[r:TRANSACTION {project_id: 'paysim_fraud_detection'}]->()
            RETURN a.account_id AS account_id, count(r) AS value
            """
        )
        in_weight = node_metric(
            """
            MATCH (a:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            OPTIONAL MATCH ()-[r:TRANSACTION {project_id: 'paysim_fraud_detection'}]->(a)
            RETURN a.account_id AS account_id, coalesce(sum(r.amount), 0.0) AS value
            """
        )
        out_weight = node_metric(
            """
            MATCH (a:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            OPTIONAL MATCH (a)-[r:TRANSACTION {project_id: 'paysim_fraud_detection'}]->()
            RETURN a.account_id AS account_id, coalesce(sum(r.amount), 0.0) AS value
            """
        )
        counterparties = node_metric(
            """
            MATCH (a:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            OPTIONAL MATCH (a)-[:TRANSACTION {project_id: 'paysim_fraud_detection'}]-
                          (b:PaySimFraudAccount {project_id: 'paysim_fraud_detection'})
            RETURN a.account_id AS account_id, count(DISTINCT b) AS value
            """
        )

    accounts = sorted(set(in_degree) | set(out_degree))
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
                counterparties.get(account, 0) for account in accounts
            ],
            "community_id": [communities.get(account, -1) for account in accounts],
        }
    )


def attach_account_features(
    transactions: pd.DataFrame,
    account_features: pd.DataFrame,
    train_end_step: int | None = None,
) -> pd.DataFrame:
    if account_features["account_id"].duplicated().any():
        raise ValueError("Account graph features must contain unique account IDs")
    indexed = account_features.set_index("account_id")
    features = pd.DataFrame(index=transactions.index)
    known_accounts = set(indexed.index)
    for role, identifier in (("source", "nameOrig"), ("destination", "nameDest")):
        ids = transactions[identifier].astype(str)
        known = ids.isin(known_accounts)
        features[f"{role}_unseen"] = (~known).astype("int8")
        for column in (
            "pagerank",
            "in_degree",
            "out_degree",
            "weighted_degree",
            "unique_counterparty",
            "community_id",
        ):
            mapped = ids.map(indexed[column])
            default = -1 if column == "community_id" else 0
            features[f"{role}_{column}"] = mapped.fillna(default).to_numpy()

    if train_end_step is not None:
        train_rows = transactions["step"] <= train_end_step
        if not train_rows.any():
            raise ValueError("No transaction rows fall within the training cutoff")
        training_transactions = transactions.loc[train_rows]
        causal = causal_static_prefix_features(training_transactions, train_end_step)
        for role in ("source", "destination"):
            for name in (
                "in_degree",
                "out_degree",
                "weighted_degree",
                "unique_counterparty",
                "unseen",
            ):
                features.loc[train_rows, f"{role}_{name}"] = causal[
                    f"{role}_{name}"
                ].to_numpy()
            features.loc[train_rows, f"{role}_pagerank"] = np.float32(0.0)
            features.loc[train_rows, f"{role}_community_id"] = -1
    return features
