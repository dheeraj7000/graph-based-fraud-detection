import numpy as np
import pandas as pd

from src.data.preprocess import make_tabular_features
from src.data.split import chronological_split
from src.evaluation.experiments import _publish_readme_results
from src.evaluation.metrics import evaluate_with_validation_threshold, select_threshold
from src.graph.static_features import (
    causal_unique_counterparty_counts,
    causal_static_prefix_features,
    training_graph_transactions,
)
from src.graph.sampled_features import (
    attach_sampled_graph_features,
    sample_training_edges,
    sampled_graph_account_features,
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
    sampled_a = sample_training_edges(transactions, train_end_step=2, max_edges=2, seed=7)
    sampled_b = sample_training_edges(transactions, train_end_step=2, max_edges=2, seed=7)
    assert len(sampled_a) == 2
    assert sampled_a.index.tolist() == sampled_b.index.tolist()
    assert sampled_a["step"].max() <= 2

    account_features = sampled_graph_account_features(sampled_a)
    row_features = attach_sampled_graph_features(
        transactions, sampled_a, account_features, train_end_step=2
    )
    assert row_features.loc[sampled_a.index, "source_pagerank"].eq(0).all()
    assert row_features.loc[sampled_a.index, "source_community_id"].eq(-1).all()
    assert row_features.loc[3, "destination_pagerank"] > 0


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


def test_threshold_selection_rejects_validation_without_both_classes() -> None:
    try:
        select_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("Expected a clear error for one-class validation labels")


def test_readme_results_publish_only_measured_rows(tmp_path, monkeypatch) -> None:
    (tmp_path / "results" / "metrics").mkdir(parents=True)
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
    (tmp_path / "results" / "metrics" / "eda_summary.json").write_text(
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
    assert "not full-graph Louvain communities" in readme
    _publish_readme_results(results)
    updated_readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert updated_readme == readme
