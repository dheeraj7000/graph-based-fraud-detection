import numpy as np
import pandas as pd

from src.data.preprocess import make_tabular_features
from src.data.split import chronological_split
from src.evaluation.metrics import evaluate_with_validation_threshold, select_threshold
from src.graph.static_features import (
    attach_account_features,
    causal_unique_counterparty_counts,
    training_graph_transactions,
)
from src.graph.temporal_features import (
    build_temporal_features,
    max_history_step_for_transaction,
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


def test_training_graph_is_cut_off_at_train_end() -> None:
    training = training_graph_transactions(sample_transactions(), 2)
    assert training["step"].max() == 2
    assert len(training) == 3


def test_training_static_graph_features_use_only_prior_steps() -> None:
    transactions = sample_transactions()
    accounts = pd.DataFrame(
        {
            "account_id": ["A", "B", "C", "D", "E"],
            "pagerank": [0.2] * 5,
            "in_degree": [2] * 5,
            "out_degree": [3] * 5,
            "weighted_degree": [10.0] * 5,
            "unique_counterparty": [2] * 5,
            "community_id": [7] * 5,
        }
    )
    features = attach_account_features(transactions, accounts, train_end_step=2)
    assert features.loc[0, "source_out_degree"] == 0
    assert features.loc[2, "source_out_degree"] == 1
    assert features.loc[2, "source_pagerank"] == 0
    assert features.loc[2, "source_community_id"] == -1
    assert features.loc[3, "source_pagerank"] == 0.2


def test_static_unique_counterparties_deduplicate_bidirectional_edges() -> None:
    transactions = pd.DataFrame(
        {
            "step": [1, 2, 3],
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
