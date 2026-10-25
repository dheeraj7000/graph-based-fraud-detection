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
    """Count distinct prior neighbors per row, with all transactions at a step batched."""
    ordered = transactions.loc[:, ["step", "nameOrig", "nameDest"]].copy()
    ordered["_row_id"] = range(len(ordered))
    ordered = ordered.sort_values(["step", "_row_id"], kind="stable")
    neighbors: dict[str, set[str]] = {}
    source_counts = np.zeros(len(ordered), dtype=np.int32)
    destination_counts = np.zeros(len(ordered), dtype=np.int32)

    for step, current in ordered.groupby("step", sort=True):
        if int(step) > train_end_step:
            break
        source_ids = current["nameOrig"].astype(str)
        destination_ids = current["nameDest"].astype(str)
        row_ids = current["_row_id"].to_numpy()
        source_counts[row_ids] = [
            len(neighbors.get(account, ())) for account in source_ids
        ]
        destination_counts[row_ids] = [
            len(neighbors.get(account, ())) for account in destination_ids
        ]
        current_pairs = pd.DataFrame(
            {"source": source_ids.to_numpy(), "destination": destination_ids.to_numpy()}
        ).drop_duplicates()
        for source, destination in current_pairs.itertuples(index=False, name=None):
            neighbors.setdefault(source, set()).add(destination)
            neighbors.setdefault(destination, set()).add(source)
    return source_counts, destination_counts


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
        source_unique, destination_unique = causal_unique_counterparty_counts(
            training_transactions, train_end_step
        )
        from src.graph.temporal_features import build_temporal_features

        full_history_window = int(training_transactions["step"].max()) - int(
            training_transactions["step"].min()
        ) + 1
        causal = build_temporal_features(
            training_transactions, window=full_history_window
        )
        for role in ("source", "destination"):
            incoming_count = causal[
                f"incoming_tx_count_{full_history_window}h"
            ]
            outgoing_count = causal[
                f"outgoing_tx_count_{full_history_window}h"
            ]
            incoming_value = causal[f"incoming_value_{full_history_window}h"]
            outgoing_value = causal[f"outgoing_value_{full_history_window}h"]
            features.loc[train_rows, f"{role}_in_degree"] = incoming_count.to_numpy()
            features.loc[train_rows, f"{role}_out_degree"] = outgoing_count.to_numpy()
            features.loc[train_rows, f"{role}_weighted_degree"] = (
                incoming_value.to_numpy() + outgoing_value.to_numpy()
            )
            unique_counts = source_unique if role == "source" else destination_unique
            features.loc[train_rows, f"{role}_unique_counterparty"] = unique_counts
            features.loc[train_rows, f"{role}_unseen"] = (
                (incoming_count + outgoing_count).eq(0).astype("int8").to_numpy()
            )
            features.loc[train_rows, f"{role}_pagerank"] = np.float32(0.0)
            features.loc[train_rows, f"{role}_community_id"] = -1
    return features
