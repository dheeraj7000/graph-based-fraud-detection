import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.preprocess import make_tabular_features
from src.data.split import chronological_split
from src.evaluation.experiments import (
    MODEL_EXCLUDED_GRAPH_COLUMNS,
    _load_features,
    _publish_readme_results,
    fraud_class_gains,
    fraud_preserving_subsample,
)
from src.evaluation.metrics import (
    evaluate_predictions,
    evaluate_with_validation_threshold,
    select_threshold,
)
from src.graph.static_features import (
    causal_unique_counterparty_counts,
    causal_static_prefix_features,
    training_graph_transactions,
)
from src.graph.account_graph import (
    account_graph_features,
    attach_graph_features,
    build_graph_feature_cache,
    select_training_edges,
    weighted_pagerank,
)
from src.graph.temporal_features import (
    build_temporal_features,
    max_history_step_for_transaction,
    stream_temporal_features,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 1, 2, 25, 26],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "TRANSFER", "PAYMENT"],
            "amount": [10.0, 5.0, 7.0, 1.0, 3.0],
            "nameOrig": ["A", "C", "A", "D", "E"],
            "oldbalanceOrg": [20.0, 10.0, 10.0, 1.0, 5.0],
            "newbalanceOrig": [10.0, 5.0, 3.0, 0.0, 2.0],
            "nameDest": ["B", "B", "B", "B", "B"],
            "oldbalanceDest": [0.0, 0.0, 15.0, 22.0, 23.0],
            "newbalanceDest": [10.0, 5.0, 22.0, 23.0, 26.0],
            "isFraud": [0, 0, 1, 0, 0],
            "isFlaggedFraud": [0, 0, 1, 0, 0],
        }
    )


def test_chronological_split_uses_ordered_unique_steps() -> None:
    frame = pd.DataFrame({"step": [4, 1, 8, 3, 2, 7, 6, 5, 9, 10]})
    train, validation, test, boundaries = chronological_split(
        frame, 0.6, 0.2, 0.2
    )
    assert boundaries.train_end_step == 6
    assert boundaries.validation_end_step == 8
    assert frame.loc[train, "step"].max() < frame.loc[validation, "step"].min()
    assert frame.loc[validation, "step"].max() < frame.loc[test, "step"].min()


def test_tabular_features_exclude_ids_and_leakage_columns() -> None:
    features = make_tabular_features(sample_transactions())
    assert not {"nameOrig", "nameDest", "isFraud", "isFlaggedFraud"} & set(features)
    assert np.isfinite(features.to_numpy()).all()


def test_temporal_features_exclude_same_step_and_expire_old_history() -> None:
    frame = sample_transactions()
    features = build_temporal_features(frame, window=24)
    assert features.loc[0, "incoming_tx_count_24h"] == 0
    assert features.loc[1, "incoming_tx_count_24h"] == 0
    assert features.loc[2, "incoming_tx_count_24h"] == 2
    assert features.loc[2, "incoming_value_24h"] == 15
    assert features.loc[2, "unique_senders_24h"] == 2
    assert features.loc[4, "incoming_tx_count_24h"] == 2
    assert features.loc[4, "incoming_value_24h"] == 8
    assert max_history_step_for_transaction([1, 2, 25], 26) == 25


def test_streaming_temporal_features_preserve_step_boundaries(tmp_path) -> None:
    transactions = sample_transactions()
    csv_path = tmp_path / "transactions.csv"
    parquet_path = tmp_path / "features.parquet"
    transactions.to_csv(csv_path, index=False)
    row_count = stream_temporal_features(csv_path, parquet_path, window=24)
    streamed = pd.read_parquet(parquet_path)
    expected = build_temporal_features(transactions, window=24)
    assert row_count == len(transactions)
    assert streamed["_row_id"].tolist() == list(range(len(transactions)))
    pd.testing.assert_frame_equal(
        streamed.drop(columns="_row_id").reset_index(drop=True), expected
    )


def test_training_graph_is_cut_off_at_train_end() -> None:
    training = training_graph_transactions(sample_transactions(), 2)
    assert training["step"].max() == 2
    assert len(training) == 3


def test_sampled_training_graph_is_seeded_capped_and_causal() -> None:
    transactions = sample_transactions()
    sampled_a = select_training_edges(transactions, train_end_step=2, max_edges=2, seed=7)
    sampled_b = select_training_edges(transactions, train_end_step=2, max_edges=2, seed=7)
    assert len(sampled_a) == 2
    assert sampled_a.index.tolist() == sampled_b.index.tolist()
    assert sampled_a["step"].max() <= 2

    account_features = account_graph_features(sampled_a)
    row_features = attach_graph_features(
        transactions, sampled_a, account_features, train_end_step=2
    )
    assert row_features.loc[sampled_a.index, "source_pagerank"].eq(0).all()
    assert row_features.loc[sampled_a.index, "source_community_id"].eq(-1).all()
    assert row_features.loc[3, "destination_pagerank"] > 0


def test_full_graph_uses_every_training_edge_and_no_future_edges() -> None:
    transactions = sample_transactions()
    full = select_training_edges(transactions, train_end_step=2, max_edges=None, seed=7)
    assert full.index.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        select_training_edges(transactions, train_end_step=2, max_edges=0, seed=7)


def random_transactions(seed: int, rows: int = 400, accounts: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = np.array([f"C{index}" for index in range(accounts)])
    return pd.DataFrame(
        {
            "step": np.sort(rng.integers(1, 30, size=rows)),
            "amount": rng.choice([0.0, 1.5, 10.0, 250.0], size=rows),
            "nameOrig": pd.Categorical(rng.choice(names, size=rows)),
            "nameDest": pd.Categorical(rng.choice(names[: accounts // 2], size=rows)),
            "isFraud": rng.integers(0, 2, size=rows),
        }
    )


def test_account_graph_features_match_networkx_reference() -> None:
    edges = random_transactions(seed=3)
    features = account_graph_features(edges).set_index("account_id")

    graph = nx.DiGraph()
    for source, destination, amount in edges[["nameOrig", "nameDest", "amount"]].itertuples(
        index=False
    ):
        previous = graph.get_edge_data(source, destination, {}).get("weight", 0.0)
        graph.add_edge(source, destination, weight=previous + amount)
    expected_rank = nx.pagerank(graph, weight="weight", tol=1e-13, max_iter=1000)
    assert set(features.index) == set(graph.nodes)
    for account in graph.nodes:
        assert features.loc[account, "pagerank"] == pytest.approx(
            expected_rank[account], abs=1e-9
        )
        assert features.loc[account, "in_degree"] == int(
            (edges["nameDest"] == account).sum()
        )
        assert features.loc[account, "out_degree"] == int(
            (edges["nameOrig"] == account).sum()
        )
        assert features.loc[account, "unique_counterparty"] == len(
            set(graph.successors(account)) | set(graph.predecessors(account))
        )
    for component in nx.weakly_connected_components(graph):
        assert features.loc[list(component), "community_id"].nunique() == 1
    assert features["community_id"].nunique() == nx.number_weakly_connected_components(
        graph
    )


def test_weighted_pagerank_redistributes_dangling_rank() -> None:
    rank = weighted_pagerank(
        np.array([0, 0]), np.array([1, 2]), np.array([1.0, 3.0]), node_count=3
    )
    expected = nx.pagerank(
        nx.DiGraph([(0, 1, {"weight": 1.0}), (0, 2, {"weight": 3.0})]),
        weight="weight",
        tol=1e-13,
    )
    assert rank.sum() == pytest.approx(1.0)
    assert rank.tolist() == pytest.approx([expected[node] for node in range(3)], abs=1e-9)


def reference_prefix_features(transactions: pd.DataFrame, train_end_step: int) -> dict:
    """Brute-force prior-step account history for every row."""
    expected: dict[str, list[float]] = {}
    for role, identifier in (("source", "nameOrig"), ("destination", "nameDest")):
        for metric in (
            "in_degree",
            "out_degree",
            "weighted_degree",
            "unique_counterparty",
            "unseen",
        ):
            expected[f"{role}_{metric}"] = []
        for row in transactions.itertuples():
            account = getattr(row, identifier)
            prior = transactions.loc[
                (transactions["step"] < row.step)
                & (transactions["step"] <= train_end_step)
            ]
            if row.step > train_end_step:
                values = (0, 0, 0.0, 0, 0)
            else:
                incoming = prior.loc[prior["nameDest"] == account]
                outgoing = prior.loc[prior["nameOrig"] == account]
                counterparties = set(incoming["nameOrig"]) | set(outgoing["nameDest"])
                values = (
                    len(incoming),
                    len(outgoing),
                    float(incoming["amount"].sum() + outgoing["amount"].sum()),
                    len(counterparties),
                    int(incoming.empty and outgoing.empty),
                )
            for metric, value in zip(
                (
                    "in_degree",
                    "out_degree",
                    "weighted_degree",
                    "unique_counterparty",
                    "unseen",
                ),
                values,
            ):
                expected[f"{role}_{metric}"].append(value)
    return expected


def test_causal_prefix_matches_brute_force_reference() -> None:
    transactions = random_transactions(seed=11, rows=250, accounts=25)
    transactions.index = transactions.index + 100
    features = causal_static_prefix_features(transactions, train_end_step=20)
    expected = reference_prefix_features(transactions, train_end_step=20)
    assert features.index.equals(transactions.index)
    for column, values in expected.items():
        assert features[column].tolist() == pytest.approx(values), column


def test_training_graph_prefix_features_use_only_prior_steps() -> None:
    transactions = sample_transactions()
    features = causal_static_prefix_features(transactions, train_end_step=2)
    assert features.loc[0, "source_out_degree"] == 0
    assert features.loc[2, "source_out_degree"] == 1
    assert features.loc[2, "destination_in_degree"] == 2
    assert features.loc[0, "source_unseen"] == 1
    assert features.loc[2, "source_unseen"] == 0


def test_static_unique_counterparties_deduplicate_bidirectional_edges() -> None:
    transactions = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "amount": [1.0, 2.0, 3.0],
            "nameOrig": ["A", "B", "A"],
            "nameDest": ["B", "A", "B"],
        }
    )
    source_counts, destination_counts = causal_unique_counterparty_counts(
        transactions, train_end_step=3
    )
    assert source_counts.tolist() == [0, 1, 1]
    assert destination_counts.tolist() == [0, 1, 1]


def test_test_labels_do_not_change_validation_threshold() -> None:
    y_validation = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.1, 0.2, 0.8, 0.9])
    test_scores = np.array([0.3, 0.7])
    threshold_a, _ = evaluate_with_validation_threshold(
        y_validation, validation_scores, np.array([0, 1]), test_scores
    )
    threshold_b, _ = evaluate_with_validation_threshold(
        y_validation, validation_scores, np.array([1, 0]), test_scores
    )
    assert threshold_a == threshold_b


def test_threshold_metrics_score_only_the_minority_fraud_class() -> None:
    y_true = np.array([0] * 995 + [1] * 5)
    scores = np.zeros(len(y_true))
    scores[[995, 996, 0, 1]] = 0.9
    metrics = evaluate_predictions(y_true, scores, threshold=0.5)
    predicted = (scores >= 0.5).astype(int)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.4)
    assert metrics["f1"] == pytest.approx(
        f1_score(y_true, predicted, pos_label=1, average="binary")
    )
    assert metrics["precision"] != pytest.approx(
        precision_score(y_true, predicted, average="weighted")
    )
    assert metrics["recall"] != pytest.approx(
        recall_score(y_true, predicted, average="macro")
    )
    assert metrics["fraud_support"] == 5
    assert (metrics["true_positives"], metrics["false_positives"]) == (2, 2)
    assert metrics["false_negatives"] == 3


def test_metrics_reject_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        evaluate_predictions(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]), 0.5)


def test_threshold_selection_rejects_validation_without_both_classes() -> None:
    try:
        select_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("Expected a clear error for one-class validation labels")


def test_readme_results_publish_only_measured_rows(tmp_path, monkeypatch) -> None:
    (tmp_path / "evaluations" / "metrics").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "Header\n<!-- BEGIN MEASURED RESULTS -->\n<!-- END MEASURED RESULTS -->\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed" / "split.json").write_text(
        '{"train_end_step": 520, "validation_end_step": 631, "test_end_step": 743, '
        '"train_steps": 600, "validation_steps": 200, "test_steps": 100}',
        encoding="utf-8",
    )
    (tmp_path / "evaluations" / "metrics" / "eda_summary.json").write_text(
        '{"transactions": 900, "fraud_count": 9, "fraud_percentage": 1.0}',
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed" / "sampled_graph_metadata.json").write_text(
        '{"sampled_training_edges": 50, "sampled_training_frauds": 2, '
        '"sampled_training_accounts": 90, "seed": 42}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    results = pd.DataFrame(
        [
            {
                "model": "Logistic",
                "features": "tabular_sampled",
                "pr_auc": 0.2,
                "precision": 0.3,
                "recall": 0.4,
                "f1": 0.34,
                "roc_auc": 0.7,
            },
            {
                "model": "Logistic",
                "features": "temporal_only",
                "pr_auc": 0.25,
                "precision": 0.35,
                "recall": 0.45,
                "f1": 0.39,
                "roc_auc": 0.75,
            },
            {
                "model": "Logistic",
                "features": "sampled_graph",
                "pr_auc": 0.3,
                "precision": 0.4,
                "recall": 0.5,
                "f1": 0.44,
                "roc_auc": 0.8,
            },
            {
                "model": "Logistic",
                "features": "sampled_temporal",
                "pr_auc": 0.35,
                "precision": 0.45,
                "recall": 0.55,
                "f1": 0.49,
                "roc_auc": 0.85,
            },
        ]
    )
    _publish_readme_results(results)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Partial local custom-graph run: 3/12 model/feature experiments completed." in readme
    assert "| Logistic | tabular_sampled | 0.200000 | 0.300000 |" in readme
    assert "Supplemental temporal-only comparison" in readme
    assert "do not count toward the planned 12-run matrix" in readme
    assert "| Logistic | 0.250000 | 0.350000 |" in readme
    assert "| Logistic | sampled_temporal | 0.350000 |" in readme
    assert "50 training transactions, including 2 fraud rows and 90 accounts" in readme
    assert "are computed for analysis but are not model inputs" in readme
    assert "Fraud precision | Fraud recall | Fraud F1" in readme
    assert "minority fraud class only" in readme
    _publish_readme_results(results)
    updated_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert updated_readme == readme


def test_full_graph_cache_covers_all_training_edges(tmp_path) -> None:
    transactions = sample_transactions()
    metadata = build_graph_feature_cache(
        transactions,
        train_end_step=2,
        kind="full",
        seed=7,
        feature_dir=tmp_path / "features",
        processed_dir=tmp_path / "processed",
    )
    assert metadata["full_training_edges"] == 3
    assert metadata["full_training_frauds"] == 1
    assert metadata["full_training_accounts"] == 3
    assert metadata["full_weak_components"] == 1
    features = pd.read_parquet(tmp_path / "features" / "full_graph_features.parquet")
    assert features["_row_id"].tolist() == list(range(len(transactions)))
    assert features.loc[2, "source_out_degree"] == 1
    assert features.loc[3, "destination_in_degree"] == 3
    assert features.loc[3, "source_unseen"] == 1
    assert not (tmp_path / "processed" / "sampled_graph_train_rows.parquet").exists()
    saved = json.loads(
        (tmp_path / "processed" / "full_graph_metadata.json").read_text(encoding="utf-8")
    )
    assert saved == metadata
    with pytest.raises(ValueError):
        build_graph_feature_cache(transactions, 2, kind="full", seed=7, edge_limit=10)


def test_readme_publishes_full_graph_fraud_class_gains(tmp_path, monkeypatch) -> None:
    (tmp_path / "evaluations" / "metrics").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "<!-- BEGIN MEASURED RESULTS -->\n<!-- END MEASURED RESULTS -->\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed" / "split.json").write_text(
        '{"train_end_step": 520, "validation_end_step": 631, "test_end_step": 743, '
        '"train_steps": 600, "validation_steps": 200, "test_steps": 100}',
        encoding="utf-8",
    )
    (tmp_path / "evaluations" / "metrics" / "eda_summary.json").write_text(
        '{"transactions": 900, "fraud_count": 9, "fraud_percentage": 1.0}',
        encoding="utf-8",
    )
    (tmp_path / "data" / "processed" / "full_graph_metadata.json").write_text(
        '{"full_training_edges": 600, "full_training_frauds": 6, '
        '"full_training_accounts": 1100, "full_weak_components": 500}',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    results = pd.DataFrame(
        [
            {"model": "XGBoost", "features": "tabular", "pr_auc": 0.5,
             "precision": 0.6, "recall": 0.4, "f1": 0.48, "roc_auc": 0.9},
            {"model": "XGBoost", "features": "full_graph", "pr_auc": 0.7,
             "precision": 0.65, "recall": 0.6, "f1": 0.624, "roc_auc": 0.95},
        ]
    )
    gains = fraud_class_gains(results)
    assert gains["comparison"].tolist() == ["full_graph_gain"]
    assert gains.loc[0, "recall_gain"] == pytest.approx(0.2)

    _publish_readme_results(results)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Full-graph experiments (every training edge)" in readme
    assert "Partial full-graph run: 2/16" in readme
    assert "all 600 training transactions, including 6 fraud rows" in readme
    assert "| XGBoost | full_graph | 0.700000 | 0.650000 | 0.600000 |" in readme
    assert (
        "| XGBoost | `full_graph` minus `tabular` | +0.200000 | +0.050000 | "
        "+0.200000 | +0.144000 |" in readme
    )
    assert "Supplemental full-training-data tabular baselines" not in readme


def test_fraud_preserving_subsample_keeps_every_fraud_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random(20_000) < 0.01).astype(np.int8)
    eligible = np.arange(20_000) < 15_000

    subset = fraud_preserving_subsample(labels, eligible, 0.1, seed=42)

    assert not subset[~eligible].any()
    assert subset[eligible & (labels == 1)].all()
    legitimate_share = subset[eligible & (labels == 0)].mean()
    assert 0.08 < legitimate_share < 0.12
    np.testing.assert_array_equal(
        subset, fraud_preserving_subsample(labels, eligible, 0.1, seed=42)
    )
    with pytest.raises(ValueError):
        fraud_preserving_subsample(labels, eligible, 0.0, seed=42)


def test_model_features_exclude_columns_constant_on_training_rows(tmp_path, monkeypatch) -> None:
    feature_dir = tmp_path / "data" / "features"
    feature_dir.mkdir(parents=True)
    row_ids = np.arange(4)
    pd.DataFrame({"_row_id": row_ids, "amount": [1.0, 2.0, 3.0, 4.0]}).to_parquet(
        feature_dir / "tabular_features.parquet", index=False
    )
    graph = {"_row_id": row_ids}
    for role in ("source", "destination"):
        for metric in ("pagerank", "community_id", "in_degree", "unseen"):
            graph[f"{role}_{metric}"] = [0.0, 1.0, 2.0, 3.0]
    pd.DataFrame(graph).to_parquet(feature_dir / "full_graph_features.parquet", index=False)
    monkeypatch.chdir(tmp_path)

    features = _load_features("full_graph")

    assert not set(MODEL_EXCLUDED_GRAPH_COLUMNS) & set(features.columns)
    assert {"source_in_degree", "destination_unseen", "amount"} <= set(features.columns)
