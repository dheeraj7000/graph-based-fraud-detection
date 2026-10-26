from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_predictions, select_threshold
from src.models.mlp import MLPTrainer
from src.utils.config import load_config

FEATURE_FILES = {
    "tabular": "tabular_features.parquet",
    "static": "static_graph_features.parquet",
    "temporal": "temporal_graph_features.parquet",
}


def _load_features(groups: tuple[str, ...]) -> pd.DataFrame:
    base = pd.read_parquet(Path("data/features") / FEATURE_FILES["tabular"])
    row_ids = base.pop("_row_id").to_numpy()
    feature_blocks = [base]
    for group in groups:
        if group == "tabular":
            continue
        path = Path("data/features") / FEATURE_FILES[group]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {group} feature cache: {path}. Run its feature-generation stage."
            )
        extra = pd.read_parquet(path)
        extra_row_ids = extra.pop("_row_id").to_numpy()
        if not np.array_equal(row_ids, extra_row_ids):
            raise ValueError(f"{group} feature rows do not match tabular row IDs")
        feature_blocks.append(extra)
    features = pd.concat(feature_blocks, axis=1, copy=False)
    if features.isna().any().any():
        raise ValueError("Feature matrix has missing values after cache merge")
    return features


def _make_classical_models(
    config: dict[str, Any], y_train: np.ndarray
) -> dict[str, Any]:
    seed = int(config["seed"])
    models: dict[str, Any] = {}
    model_config = config["models"]
    if model_config.get("logistic", True):
        models["Logistic"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                solver="saga",
                max_iter=200,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if model_config.get("random_forest", True):
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=int(model_config.get("random_forest_estimators", 200)),
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if model_config.get("xgboost", True):
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is enabled but not installed; install requirements.txt"
            ) from error
        positives = int(np.sum(y_train))
        negatives = len(y_train) - positives
        if positives == 0:
            raise ValueError("Training partition has no fraud examples")
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negatives / positives,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    return models


def _save_plots(
    results: pd.DataFrame,
    curves: list[tuple[np.ndarray, np.ndarray, str]],
) -> None:
    figure_dir = Path("results/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    if curves:
        fig, ax = plt.subplots(figsize=(8, 6))
    for y_test, scores, label in curves:
        display = PrecisionRecallDisplay.from_predictions(
            y_test, scores, name=label, plot_chance_level=False, ax=ax
        )
    if curves:
        ax.set_title("Test precision-recall curves")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / "precision_recall_curves.png", dpi=160)
        plt.close(fig)

    pivot = results.pivot(index="model", columns="features", values="pr_auc")
    ax = pivot.plot.bar(figsize=(10, 6))
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Fraud detection PR-AUC by model and feature set")
    plt.tight_layout()
    plt.savefig(figure_dir / "model_pr_auc_comparison.png", dpi=160)
    plt.close()

    gains = results.pivot(index="model", columns="features", values="pr_auc")
    ablation = pd.DataFrame(index=gains.index)
    if {"tabular", "static"}.issubset(gains.columns):
        ablation["static_graph_gain"] = gains["static"] - gains["tabular"]
    if {"static", "temporal"}.issubset(gains.columns):
        ablation["temporal_gain"] = gains["temporal"] - gains["static"]
    if {"tabular", "temporal"}.issubset(gains.columns):
        ablation["total_graph_gain"] = gains["temporal"] - gains["tabular"]
    if not ablation.empty:
        ablation.to_csv("results/metrics/ablation_gains.csv")
        ax = ablation.plot.bar(figsize=(10, 6))
        ax.set_ylabel("PR-AUC gain")
        ax.set_title("Feature ablation gains")
        plt.tight_layout()
        plt.savefig(figure_dir / "ablation_results.png", dpi=160)
        plt.close()


def _save_research_summary(results: pd.DataFrame) -> None:
    expected = {
        (model, feature_set)
        for model in ("Logistic", "Random Forest", "XGBoost", "MLP")
        for feature_set in ("tabular", "static", "temporal")
    }
    completed = set(zip(results["model"], results["features"]))
    if not expected.issubset(completed):
        return
    by_model = results.pivot(index="model", columns="features")
    static_gain = by_model["pr_auc"]["static"] - by_model["pr_auc"]["tabular"]
    temporal_gain = by_model["pr_auc"]["temporal"] - by_model["pr_auc"]["static"]
    total_gain = by_model["pr_auc"]["temporal"] - by_model["pr_auc"]["tabular"]
    best_model = str(total_gain.idxmax())
    precision_delta = (
        by_model["precision"]["temporal"] - by_model["precision"]["tabular"]
    )
    recall_delta = by_model["recall"]["temporal"] - by_model["recall"]["tabular"]
    precision_magnitude = float(precision_delta.abs().mean())
    recall_magnitude = float(recall_delta.abs().mean())
    if np.isclose(precision_magnitude, recall_magnitude):
        primary_change = "precision and recall to a similar degree"
    elif precision_magnitude > recall_magnitude:
        primary_change = "precision more than recall"
    else:
        primary_change = "recall more than precision"

    lines = [
        "# Main experiment findings",
        "",
        "All PR-AUC values and gains below are measured on the untouched chronological test set. "
        "They describe association, not causation.",
        "",
        f"1. **Do static graph features improve fraud detection?** Mean static-minus-tabular "
        f"PR-AUC across model families: {static_gain.mean():+.6f} "
        f"({int((static_gain > 0).sum())}/{len(static_gain)} models improved).",
        f"2. **Does recent network activity add signal beyond static structure?** Mean "
        f"temporal-minus-static PR-AUC: {temporal_gain.mean():+.6f} "
        f"({int((temporal_gain > 0).sum())}/{len(temporal_gain)} models improved).",
        f"3. **Which model benefits most from graph features?** {best_model} has the largest "
        f"temporal-minus-tabular PR-AUC gain ({total_gain[best_model]:+.6f}).",
        f"4. **Do graph features primarily improve precision, recall, or both?** For "
        f"temporal versus tabular predictions at their validation-selected thresholds, "
        f"mean precision change is {precision_delta.mean():+.6f} and mean recall change is "
        f"{recall_delta.mean():+.6f}; absolute changes are larger for {primary_change}.",
        "5. **How does performance change for transactions involving unseen accounts?** "
        "See `unseen_account_rates.csv` for validation/test coverage and "
        "`best_model_unseen_subgroups.csv` for the best model's known/unseen source-destination breakdown.",
        "",
        "Per-model results and ablation gains are in `main_results.csv` and "
        "`ablation_gains.csv`. No outcome is assumed in advance.",
    ]
    Path("results/metrics/research_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _publish_readme_results(results: pd.DataFrame) -> None:
    readme_path = Path("README.md")
    if not readme_path.is_file():
        raise FileNotFoundError("README.md is required to publish experiment results")
    readme = readme_path.read_text(encoding="utf-8")
    start_marker = "<!-- BEGIN MEASURED RESULTS -->"
    end_marker = "<!-- END MEASURED RESULTS -->"
    if readme.count(start_marker) != 1 or readme.count(end_marker) != 1:
        raise ValueError("README.md must contain exactly one measured-results marker pair")

    split_path = Path("data/processed/split.json")
    eda_path = Path("results/metrics/eda_summary.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eda = json.loads(eda_path.read_text(encoding="utf-8"))
    rows = {
        (str(row["model"]), str(row["features"])): row
        for row in results.to_dict(orient="records")
    }
    model_order = ("Logistic", "Random Forest", "XGBoost", "MLP")
    feature_order = ("tabular", "static", "temporal")
    table = [
        "| Model | Features | PR-AUC | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in model_order:
        for feature_set in feature_order:
            record = rows.get((model, feature_set))
            if record is None:
                values = ["Not run"] * 5
            else:
                values = [
                    f"{float(record[key]):.6f}"
                    for key in ("pr_auc", "precision", "recall", "f1", "roc_auc")
                ]
            table.append(
                f"| {model} | {feature_set} | "
                + " | ".join(values)
                + " |"
            )

    completed_groups = sorted(results["features"].unique().tolist())
    expected_pairs = {
        (model, feature_set)
        for model in model_order
        for feature_set in feature_order
    }
    completed_pairs = set(rows)
    if expected_pairs.issubset(completed_pairs):
        status = "All 12 planned model/feature experiments completed."
    else:
        status = (
            f"Partial run: {len(completed_pairs)}/12 planned experiments completed. "
            "Results below are measured; missing matrix cells are not run. "
            "Do not interpret the table as a completed ablation."
        )
    section = "\n".join(
        [
            "## Measured experiment results",
            "",
            f"**Status:** {status}",
            "",
            f"- Dataset: {int(eda['transactions']):,} transactions; "
            f"{int(eda['fraud_count']):,} fraud "
            f"({float(eda['fraud_percentage']):.4f}%).",
            f"- Chronological boundaries by unique step: train through "
            f"{int(split['train_end_step'])}, validation through "
            f"{int(split['validation_end_step'])}, test through "
            f"{int(split['test_end_step'])}. Rows: "
            f"{int(split['train_steps']):,} train / "
            f"{int(split['validation_steps']):,} validation / "
            f"{int(split['test_steps']):,} test.",
            "- Reported PR-AUC, ROC-AUC, precision, recall, and F1 are test-set measurements. "
            "Threshold-dependent metrics use thresholds chosen on validation data only.",
            "",
            *table,
            "",
            "The PaySim CSV was obtained from [Kaggle: Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. "
            "The raw dataset is not included in this repository.",
            "",
        ]
    )
    readme_path.write_text(
        readme.replace(
            f"{start_marker}\n{end_marker}",
            f"{start_marker}\n{section}{end_marker}",
        ),
        encoding="utf-8",
    )


def run_experiments(
    feature_groups: tuple[str, ...] = ("tabular", "static", "temporal")
) -> pd.DataFrame:
    config = load_config()
    metadata_path = Path("data/processed/split_metadata.parquet")
    if not metadata_path.is_file():
        raise FileNotFoundError("Run scripts/run_preprocessing.py first")
    metadata = pd.read_parquet(metadata_path)
    tabular_file = Path("data/features") / FEATURE_FILES["tabular"]
    tabular_columns = [
        column
        for column in pq.ParquetFile(tabular_file).schema.names
        if column != "_row_id"
    ]
    static_path = Path("data/features") / FEATURE_FILES["static"]
    temporal_path = Path("data/features") / FEATURE_FILES["temporal"]
    static_columns = (
        [
            column
            for column in pq.ParquetFile(static_path).schema.names
            if column != "_row_id"
        ]
        if static_path.is_file()
        else []
    )
    temporal_columns = (
        [
            column
            for column in pq.ParquetFile(temporal_path).schema.names
            if column != "_row_id"
        ]
        if temporal_path.is_file()
        else []
    )
    row_ids = metadata["_row_id"].to_numpy()
    if not np.array_equal(row_ids, np.arange(len(metadata))):
        raise ValueError("Split metadata row IDs must be contiguous and ordered")
    split = metadata["split"].to_numpy()
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("Training, validation, and test partitions must all be non-empty")
    y = metadata["isFraud"].to_numpy(dtype=np.int8)
    if np.unique(y[validation]).size != 2:
        raise ValueError("Validation period must contain fraud and non-fraud rows")
    if np.unique(y[test]).size != 2:
        raise ValueError("Test period must contain fraud and non-fraud rows")
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    prediction_dir = Path("results/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    for group in feature_groups:
        features = _load_features((group,))
        if group == "tabular":
            selected = tabular_columns
        elif group == "static":
            selected = tabular_columns + static_columns
        else:
            selected = tabular_columns + static_columns + temporal_columns
        group_x = features[selected].to_numpy(dtype=np.float32)
        classical = _make_classical_models(config, y[train])
        models: dict[str, Any] = dict(classical)
        if config["models"].get("mlp", True):
            models["MLP"] = MLPTrainer(
                seed=int(config["seed"]),
                epochs=int(config["models"].get("mlp_epochs", 30)),
                batch_size=int(config["models"].get("mlp_batch_size", 8192)),
                patience=int(config["models"].get("early_stopping_patience", 5)),
            )

        for model_name, model in models.items():
            if model_name == "MLP":
                model.fit(group_x[train], y[train], group_x[validation], y[validation])
                validation_scores = model.predict_proba(group_x[validation])
                test_scores = model.predict_proba(group_x[test])
            else:
                model.fit(group_x[train], y[train])
                validation_scores = model.predict_proba(group_x[validation])[:, 1]
                test_scores = model.predict_proba(group_x[test])[:, 1]
            threshold = select_threshold(y[validation], validation_scores)
            metrics = evaluate_predictions(y[test], test_scores, threshold)
            results.append(
                {
                    "model": model_name,
                    "features": group,
                    **metrics,
                    "train_rows": int(train.sum()),
                    "validation_rows": int(validation.sum()),
                    "test_rows": int(test.sum()),
                }
            )
            key = f"{model_name}_{group}"
            test_ids = row_ids[test]
            pd.DataFrame(
                {
                    "_row_id": test_ids,
                    "y_true": y[test],
                    "probability": test_scores,
                    "threshold": threshold,
                    "prediction": test_scores >= threshold,
                }
            ).to_parquet(prediction_dir / f"{key}.parquet", index=False)
            curves.append((y[test], test_scores, key))

            if model_name in ("XGBoost", "Random Forest"):
                importances = pd.DataFrame(
                    {"feature": selected, "importance": model.feature_importances_}
                ).sort_values("importance", ascending=False)
                importance_dir = Path("results/metrics")
                importance_dir.mkdir(parents=True, exist_ok=True)
                model_slug = model_name.lower().replace(" ", "_")
                importances.to_csv(
                    importance_dir / f"{model_slug}_importance_{group}.csv",
                    index=False,
                )
                ax = importances.head(20).sort_values("importance").plot.barh(
                    x="feature", y="importance", legend=False, figsize=(9, 7)
                )
                ax.set_title(f"{model_name} feature importance ({group})")
                plt.tight_layout()
                plt.savefig(
                    Path("results/figures")
                    / f"feature_importance_{model_slug}_{group}.png",
                    dpi=160,
                )
                plt.close()

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "main_results.csv", index=False)
    _save_plots(results_frame, curves)
    _publish_readme_results(results_frame)
    _save_research_summary(results_frame)
    static_path = Path("data/features") / FEATURE_FILES["static"]
    if static_path.is_file() and not results_frame.empty:
        static = pd.read_parquet(static_path)
        unseen_summary = []
        for period, mask in (("validation", validation), ("test", test)):
            period_static = static.loc[mask]
            unseen_summary.append(
                {
                    "period": period,
                    "source_unseen_percent": float(
                        period_static["source_unseen"].mean() * 100
                    ),
                    "destination_unseen_percent": float(
                        period_static["destination_unseen"].mean() * 100
                    ),
                }
            )
        pd.DataFrame(unseen_summary).to_csv(
            output_dir / "unseen_account_rates.csv", index=False
        )

        best = results_frame.loc[results_frame["pr_auc"].idxmax()]
        prediction_path = prediction_dir / f"{best['model']}_{best['features']}.parquet"
        best_predictions = pd.read_parquet(prediction_path)
        test_static = static.loc[test].reset_index(drop=True)
        source_unseen = test_static["source_unseen"].astype(bool).to_numpy()
        destination_unseen = test_static["destination_unseen"].astype(bool).to_numpy()
        categories = {
            "known_to_known": ~source_unseen & ~destination_unseen,
            "known_to_unseen": ~source_unseen & destination_unseen,
            "unseen_to_known": source_unseen & ~destination_unseen,
            "unseen_to_unseen": source_unseen & destination_unseen,
        }
        subgroup_rows = []
        y_test = best_predictions["y_true"].to_numpy()
        scores = best_predictions["probability"].to_numpy()
        predictions = best_predictions["prediction"].astype(bool).to_numpy()
        for category, mask in categories.items():
            if not mask.any():
                subgroup_rows.append({"category": category, "transactions": 0})
                continue
            labels = y_test[mask]
            selected_scores = scores[mask]
            selected_predictions = predictions[mask]
            subgroup_rows.append(
                {
                    "category": category,
                    "transactions": int(mask.sum()),
                    "positive_transactions": int(labels.sum()),
                    "pr_auc": float(average_precision_score(labels, selected_scores))
                    if labels.sum()
                    else float("nan"),
                    "roc_auc": float(roc_auc_score(labels, selected_scores))
                    if np.unique(labels).size == 2
                    else float("nan"),
                    "precision": float(
                        precision_score(labels, selected_predictions, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(labels, selected_predictions, zero_division=0)
                    ),
                    "f1": float(f1_score(labels, selected_predictions, zero_division=0)),
                }
            )
        pd.DataFrame(subgroup_rows).to_csv(
            output_dir / "best_model_unseen_subgroups.csv", index=False
        )
    print(results_frame[["model", "features", "pr_auc", "precision", "recall", "f1"]])
    return results_frame
