from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_predictions, select_threshold
from src.models.mlp import MLPTrainer
from src.utils.config import load_config

FEATURE_FILES = {
    "tabular": "tabular_features.parquet",
    "temporal": "temporal_graph_features.parquet",
    "sampled_graph": "sampled_graph_features.parquet",
    "full_graph": "full_graph_features.parquet",
}
CACHE_STAGES = {
    "temporal": "code/scripts/run_temporal_features.py",
    "sampled_graph": "code/scripts/run_sampled_graph_features.py",
    "full_graph": "code/scripts/run_full_graph_features.py",
}
FEATURE_GROUP_BLOCKS: dict[str, tuple[str, ...]] = {
    "tabular": (),
    "tabular_sampled": (),
    "sampled_graph": ("sampled_graph",),
    "sampled_temporal": ("sampled_graph", "temporal"),
    "temporal_only": ("temporal",),
    "full_graph": ("full_graph",),
    "full_temporal": ("full_graph", "temporal"),
}
# Global PageRank and component IDs would leak later edges into training rows, so
# training rows hold constants for them; models must not see features that are
# constant in training but vary at test time.
MODEL_EXCLUDED_GRAPH_COLUMNS = tuple(
    f"{role}_{metric}"
    for role in ("source", "destination")
    for metric in ("pagerank", "community_id")
)
SAMPLED_FEATURE_GROUPS = ("tabular_sampled", "sampled_graph", "sampled_temporal")
FULL_FEATURE_GROUPS = ("tabular", "full_graph", "temporal_only", "full_temporal")
MODEL_ORDER = ("Logistic", "Random Forest", "XGBoost", "MLP")
REPORTED_METRICS = ("pr_auc", "precision", "recall", "f1", "roc_auc")
METRIC_HEADER = "PR-AUC | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC"
FRAUD_CLASS_NOTE = (
    "Precision, recall, and F1 are computed for the minority fraud class only "
    "(`isFraud = 1`; scikit-learn `average=\"binary\"`, `pos_label=1`). "
    "Majority-class, macro-averaged, and weighted-averaged scores are never reported, "
    "because the legitimate class would mask fraud-class performance at a 0.13% fraud rate."
)
GAIN_COMPARISONS = {
    "sampled_graph_gain": ("sampled_graph", "tabular_sampled"),
    "temporal_gain": ("sampled_temporal", "sampled_graph"),
    "total_graph_gain": ("sampled_temporal", "tabular_sampled"),
    "full_graph_gain": ("full_graph", "tabular"),
    "full_graph_gain_over_temporal": ("full_temporal", "temporal_only"),
    "full_total_graph_gain": ("full_temporal", "tabular"),
}


def _load_features(group: str) -> pd.DataFrame:
    if group not in FEATURE_GROUP_BLOCKS:
        raise ValueError(f"Unknown feature group: {group}")
    base = pd.read_parquet(Path("data/features") / FEATURE_FILES["tabular"])
    row_ids = base.pop("_row_id").to_numpy()
    feature_blocks = [base]
    for block in FEATURE_GROUP_BLOCKS[group]:
        path = Path("data/features") / FEATURE_FILES[block]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing feature cache: {path}. Run {CACHE_STAGES[block]} first."
            )
        extra = pd.read_parquet(path)
        extra_row_ids = extra.pop("_row_id").to_numpy()
        if not np.array_equal(row_ids, extra_row_ids):
            raise ValueError(f"{path.name} feature rows do not match tabular row IDs")
        extra = extra.drop(columns=list(MODEL_EXCLUDED_GRAPH_COLUMNS), errors="ignore")
        feature_blocks.append(extra)
    features = pd.concat(feature_blocks, axis=1, copy=False)
    if features.columns.duplicated().any():
        raise ValueError(f"Feature group {group} has duplicate column names")
    if features.isna().any().any():
        raise ValueError("Feature matrix has missing values after cache merge")
    return features


def fraud_preserving_subsample(
    labels: np.ndarray, eligible: np.ndarray, legitimate_fraction: float, seed: int
) -> np.ndarray:
    """Keep every eligible fraud row and a seeded fraction of eligible legitimate rows."""
    if not 0 < legitimate_fraction <= 1:
        raise ValueError("legitimate_fraction must be in (0, 1]")
    keep = np.random.default_rng(seed).random(len(labels)) < legitimate_fraction
    return eligible & ((labels == 1) | keep)


def fraud_class_gains(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model metric changes between feature sets, for the fraud class."""
    indexed = results.set_index(["model", "features"])
    rows = []
    for comparison, (treatment, baseline) in GAIN_COMPARISONS.items():
        for model in MODEL_ORDER:
            if (model, treatment) not in indexed.index or (
                model,
                baseline,
            ) not in indexed.index:
                continue
            after = indexed.loc[(model, treatment)]
            before = indexed.loc[(model, baseline)]
            rows.append(
                {
                    "comparison": comparison,
                    "model": model,
                    "treatment": treatment,
                    "baseline": baseline,
                    **{
                        f"{metric}_gain": float(after[metric]) - float(before[metric])
                        for metric in ("pr_auc", "precision", "recall", "f1")
                    },
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "model",
            "treatment",
            "baseline",
            "pr_auc_gain",
            "precision_gain",
            "recall_gain",
            "f1_gain",
        ],
    )


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
    figure_dir = Path("evaluations/figures")
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

    gains = fraud_class_gains(results)
    if not gains.empty:
        gains.to_csv("evaluations/metrics/fraud_class_gains.csv", index=False)
        ablation = gains.pivot(index="model", columns="comparison", values="pr_auc_gain")
        ablation = ablation.reindex(
            columns=[name for name in GAIN_COMPARISONS if name in ablation.columns]
        )
        ablation.to_csv("evaluations/metrics/ablation_gains.csv")
        ax = ablation.plot.bar(figsize=(10, 6))
        ax.set_ylabel("PR-AUC gain")
        ax.set_title("Feature ablation gains")
        plt.tight_layout()
        plt.savefig(figure_dir / "ablation_results.png", dpi=160)
        plt.close()

        f1_gains = gains.pivot(index="model", columns="comparison", values="f1_gain")
        f1_gains = f1_gains.reindex(
            columns=[name for name in GAIN_COMPARISONS if name in f1_gains.columns]
        )
        ax = f1_gains.plot.bar(figsize=(10, 6))
        ax.set_ylabel("Fraud-class F1 gain")
        ax.set_title("Feature ablation gains for the fraud class (F1)")
        plt.tight_layout()
        plt.savefig(figure_dir / "fraud_f1_ablation_results.png", dpi=160)
        plt.close()


def _completed(results: pd.DataFrame, feature_groups: tuple[str, ...]) -> bool:
    expected = {
        (model, feature_set) for model in MODEL_ORDER for feature_set in feature_groups
    }
    return expected.issubset(set(zip(results["model"], results["features"])))


def _mean_gain_text(gains: pd.DataFrame, comparison: str) -> str:
    selected = gains.loc[gains["comparison"] == comparison]
    return (
        f"PR-AUC {selected['pr_auc_gain'].mean():+.6f}, "
        f"fraud precision {selected['precision_gain'].mean():+.6f}, "
        f"fraud recall {selected['recall_gain'].mean():+.6f}, "
        f"fraud F1 {selected['f1_gain'].mean():+.6f} "
        f"({int((selected['f1_gain'] > 0).sum())}/{len(selected)} models improved fraud F1)"
    )


def _full_graph_findings(results: pd.DataFrame) -> list[str]:
    gains = fraud_class_gains(results)
    metadata_path = Path("data/processed/full_graph_metadata.json")
    scope = "every training transaction"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scope = (
            f"all {int(metadata['full_training_edges']):,} training transactions "
            f"({int(metadata['full_training_accounts']):,} accounts)"
        )
    lines = [
        "",
        "## Full-graph findings",
        "",
        f"The full graph is built from {scope}; every model is fit on all training-period "
        "fraud rows plus a seeded subsample of legitimate training rows (the same rows for "
        "every feature set). Precision, recall, and F1 are fraud-class (minority-class) scores.",
        "",
        "6. **Do full-network structural features move the needle for the fraud class?** "
        "Mean full-graph-minus-tabular change: "
        f"{_mean_gain_text(gains, 'full_graph_gain')}.",
        "7. **Do full-graph features add fraud-class signal beyond 24-hour activity?** "
        "Mean full-temporal-minus-temporal-only change: "
        f"{_mean_gain_text(gains, 'full_graph_gain_over_temporal')}.",
    ]
    if _completed(results, SAMPLED_FEATURE_GROUPS):
        lines.append(
            "8. **Does scaling beyond the sampled graph change the conclusion?** "
            "Sampled graph-minus-tabular mean change: "
            f"{_mean_gain_text(gains, 'sampled_graph_gain')}. Full graph-minus-tabular "
            f"mean change: {_mean_gain_text(gains, 'full_graph_gain')}. The two designs "
            "also differ in training-set size, so compare each gain with its own baseline."
        )
    return lines


def _save_research_summary(results: pd.DataFrame) -> None:
    sampled_complete = _completed(results, SAMPLED_FEATURE_GROUPS)
    full_complete = _completed(results, FULL_FEATURE_GROUPS)
    if not sampled_complete and not full_complete:
        return
    lines = [
        "# Main experiment findings",
        "",
        "All values and gains below are measured on the untouched chronological test set. "
        "They describe association, not causation.",
        "",
        FRAUD_CLASS_NOTE,
    ]
    if sampled_complete:
        lines.extend(["", "## Sampled-graph findings", "", *_sampled_graph_findings(results)])
    if full_complete:
        lines.extend(_full_graph_findings(results))
    lines.extend(
        [
            "",
            "Per-model results and ablation gains are in `main_results.csv`, "
            "`ablation_gains.csv`, and `fraud_class_gains.csv`. No outcome is assumed in advance.",
        ]
    )
    Path("evaluations/metrics/research_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _sampled_graph_findings(results: pd.DataFrame) -> list[str]:
    by_model = results.pivot(index="model", columns="features")
    static_gain = (
        by_model["pr_auc"]["sampled_graph"]
        - by_model["pr_auc"]["tabular_sampled"]
    )
    temporal_gain = (
        by_model["pr_auc"]["sampled_temporal"]
        - by_model["pr_auc"]["sampled_graph"]
    )
    total_gain = (
        by_model["pr_auc"]["sampled_temporal"]
        - by_model["pr_auc"]["tabular_sampled"]
    )
    best_model = str(total_gain.idxmax())
    precision_delta = (
        by_model["precision"]["sampled_temporal"]
        - by_model["precision"]["tabular_sampled"]
    )
    recall_delta = (
        by_model["recall"]["sampled_temporal"]
        - by_model["recall"]["tabular_sampled"]
    )
    precision_magnitude = float(precision_delta.abs().mean())
    recall_magnitude = float(recall_delta.abs().mean())
    if np.isclose(precision_magnitude, recall_magnitude):
        primary_change = "precision and recall to a similar degree"
    elif precision_magnitude > recall_magnitude:
        primary_change = "precision more than recall"
    else:
        primary_change = "recall more than precision"

    return [
        f"1. **Do sampled custom-graph features improve fraud detection?** Mean "
        "sampled-graph-minus-tabular "
        f"PR-AUC across model families: {static_gain.mean():+.6f} "
        f"({int((static_gain > 0).sum())}/{len(static_gain)} models improved).",
        f"2. **Does recent activity add signal beyond sampled graph structure?** Mean "
        f"sampled-temporal-minus-sampled-graph PR-AUC: {temporal_gain.mean():+.6f} "
        f"({int((temporal_gain > 0).sum())}/{len(temporal_gain)} models improved).",
        f"3. **Which model benefits most from custom graph and temporal features?** "
        f"{best_model} has the largest sampled-temporal-minus-tabular PR-AUC gain "
        f"({total_gain[best_model]:+.6f}).",
        f"4. **Do graph features primarily improve fraud-class precision, recall, or "
        f"both?** For sampled-temporal versus sampled-tabular predictions at their "
        f"validation-selected thresholds, "
        f"mean fraud precision change is {precision_delta.mean():+.6f} and mean fraud "
        f"recall change is {recall_delta.mean():+.6f}; absolute changes are larger for "
        f"{primary_change}.",
        "5. **How should these results be interpreted?** They describe this deterministic "
        "sampled training graph and full chronological validation/test periods; "
        "see the full-graph findings for estimates from every training edge.",
    ]


def _metric_cells(record: dict[str, Any] | None) -> str:
    if record is None:
        return " | ".join(["Not run"] * len(REPORTED_METRICS))
    return " | ".join(f"{float(record[key]):.6f}" for key in REPORTED_METRICS)


def _matrix_table(
    rows: dict[tuple[str, str], dict[str, Any]], feature_groups: tuple[str, ...]
) -> list[str]:
    return [
        f"| Model | Features | {METRIC_HEADER} |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[
            f"| {model} | {feature_set} | "
            f"{_metric_cells(rows.get((model, feature_set)))} |"
            for model in MODEL_ORDER
            for feature_set in feature_groups
        ],
    ]


def _model_table(records: list[dict[str, Any]]) -> list[str]:
    return [
        f"| Model | {METRIC_HEADER} |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {record['model']} | {_metric_cells(record)} |" for record in records],
    ]


def _full_graph_section(
    results: pd.DataFrame, rows: dict[tuple[str, str], dict[str, Any]]
) -> str:
    expected = {
        (model, feature_set)
        for model in MODEL_ORDER
        for feature_set in FULL_FEATURE_GROUPS
    }
    completed = len(expected.intersection(rows))
    if completed == len(expected):
        status = f"All {len(expected)} full-graph model/feature experiments completed."
    else:
        status = (
            f"Partial full-graph run: {completed}/{len(expected)} model/feature "
            "experiments completed; missing matrix cells are not run."
        )
    metadata_path = Path("data/processed/full_graph_metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError("Full graph results require full_graph_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gains = fraud_class_gains(results)
    gains = gains.loc[
        gains["comparison"].isin(("full_graph_gain", "full_graph_gain_over_temporal"))
    ]
    lines = [
        "### Full-graph experiments (every training edge)",
        "",
        f"**Status:** {status}",
        "",
        f"The full graph uses all {int(metadata['full_training_edges']):,} training "
        f"transactions, including {int(metadata['full_training_frauds']):,} fraud rows, "
        f"{int(metadata['full_training_accounts']):,} accounts, and "
        f"{int(metadata['full_weak_components']):,} weakly connected components. Every "
        "model is fit on all training-period fraud rows plus a seeded subsample of "
        "legitimate training rows (`models.full_train_legitimate_fraction`), the same rows "
        "for every feature set; validation and test are the same full chronological "
        "partitions as above. PageRank and component IDs are not model inputs, because "
        "training rows cannot receive them without future edges. Feature sets: `tabular`, `full_graph` "
        "(tabular plus full graph), `temporal_only` (tabular plus 24-hour temporal), "
        "and `full_temporal` (tabular plus full graph and 24-hour temporal).",
        "",
        *_matrix_table(rows, FULL_FEATURE_GROUPS),
    ]
    if not gains.empty:
        lines.extend(
            [
                "",
                "Fraud-class change from adding full-graph features (treatment minus baseline):",
                "",
                "| Model | Comparison | PR-AUC | Fraud precision | Fraud recall | Fraud F1 |",
                "|---|---|---:|---:|---:|---:|",
                *[
                    f"| {gain.model} | `{gain.treatment}` minus `{gain.baseline}` | "
                    f"{gain.pr_auc_gain:+.6f} | {gain.precision_gain:+.6f} | "
                    f"{gain.recall_gain:+.6f} | {gain.f1_gain:+.6f} |"
                    for gain in gains.itertuples()
                ],
            ]
        )
    return "\n".join(lines)


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
    eda_path = Path("evaluations/metrics/eda_summary.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eda = json.loads(eda_path.read_text(encoding="utf-8"))
    rows = {
        (str(row["model"]), str(row["features"])): row
        for row in results.to_dict(orient="records")
    }
    model_order = MODEL_ORDER
    table = _matrix_table(rows, SAMPLED_FEATURE_GROUPS)

    expected_pairs = {
        (model, feature_set)
        for model in model_order
        for feature_set in SAMPLED_FEATURE_GROUPS
    }
    completed_pairs = set(rows).intersection(expected_pairs)
    if expected_pairs.issubset(completed_pairs):
        status = "All 12 local custom-graph model/feature experiments completed."
    else:
        status = (
            f"Partial local custom-graph run: {len(completed_pairs)}/12 "
            "model/feature experiments completed. "
            "Results below are measured; missing matrix cells are not run. "
            "Do not interpret the table as a completed comparison."
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
            "Threshold-dependent metrics use thresholds chosen on validation fraud-class F1 only.",
            f"- {FRAUD_CLASS_NOTE}",
            "- Sampled custom graph comparisons use the same deterministic sampled training rows "
            "for all models and feature sets. Validation and test rows remain unchanged.",
            "",
            *table,
        ]
    )
    full_graph_present = any(
        (model, feature_set) in rows
        for model in model_order
        for feature_set in ("full_graph", "full_temporal")
    )
    temporal_only_rows = [
        rows[(model, "temporal_only")]
        for model in model_order
        if (model, "temporal_only") in rows
    ]
    if temporal_only_rows and not full_graph_present:
        section += "\n\n" + "\n".join(
            [
                "### Supplemental temporal-only comparison",
                "",
                "These runs use tabular plus 24-hour temporal features, without static graph features. "
                "They are reported separately and do not count toward the planned 12-run matrix.",
                "",
                *_model_table(temporal_only_rows),
            ]
        )
    if any((model, "sampled_graph") in rows for model in model_order):
        sample_metadata_path = Path("data/processed/sampled_graph_metadata.json")
        if not sample_metadata_path.is_file():
            raise FileNotFoundError(
                "Sampled graph results require sampled_graph_metadata.json"
            )
        sample_metadata = json.loads(
            sample_metadata_path.read_text(encoding="utf-8")
        )
        section += "\n\n" + "\n".join(
            [
                "The local custom graph is built from a deterministic random "
                f"sample of {int(sample_metadata['sampled_training_edges']):,} training "
                f"transactions, including {int(sample_metadata['sampled_training_frauds']):,} "
                f"fraud rows and {int(sample_metadata['sampled_training_accounts']):,} "
                f"accounts (seed {int(sample_metadata['seed'])}).",
                "",
                "All models are fit on these sampled training rows; the three feature sets "
                "are tabular-only, tabular plus sampled graph, and tabular plus sampled graph "
                "and 24-hour temporal features. Validation and test rows remain unchanged. "
                "Graph model inputs are in/out degree, weighted degree, unique counterparties, "
                "and an unseen-account flag. Sparse weighted PageRank and weakly connected "
                "components are computed for analysis but are not model inputs, because "
                "training rows cannot receive them without future edges.",
            ]
        )
    full_tabular_rows = [
        rows[(model, "tabular")] for model in model_order if (model, "tabular") in rows
    ]
    if full_graph_present:
        section += "\n\n" + _full_graph_section(results, rows)
    elif full_tabular_rows:
        section += "\n\n" + "\n".join(
            [
                "### Supplemental full-training-data tabular baselines",
                "",
                "These previously measured tabular runs used the full training split and "
                "are reported separately; compare sampled-graph matrix cells against "
                "`tabular_sampled` for a matched training sample.",
                "",
                *_model_table(full_tabular_rows),
            ]
        )
    section += (
        "\n\nThe PaySim CSV was obtained from [Kaggle: Synthetic Financial Datasets For Fraud Detection]"
        "(https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. "
        "The raw dataset is not included in this repository.\n"
    )
    start_index = readme.index(start_marker) + len(start_marker)
    end_index = readme.index(end_marker)
    if start_index >= end_index:
        raise ValueError("README.md measured-results markers are out of order")
    readme_path.write_text(
        f"{readme[:start_index]}\n{section}{readme[end_index:]}",
        encoding="utf-8",
    )


def run_experiments(
    feature_groups: tuple[str, ...] = SAMPLED_FEATURE_GROUPS,
) -> pd.DataFrame:
    unknown = [group for group in feature_groups if group not in FEATURE_GROUP_BLOCKS]
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    config = load_config()
    metadata_path = Path("data/processed/split_metadata.parquet")
    if not metadata_path.is_file():
        raise FileNotFoundError("Run code/scripts/run_preprocessing.py first")
    metadata = pd.read_parquet(metadata_path)
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
    output_dir = Path("evaluations/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("evaluations/figures").mkdir(parents=True, exist_ok=True)
    prediction_dir = Path("evaluations/predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    for group in feature_groups:
        fit_train = train.copy()
        if group in SAMPLED_FEATURE_GROUPS:
            sampled_path = Path("data/processed/sampled_graph_train_rows.parquet")
            if not sampled_path.is_file():
                raise FileNotFoundError(
                    "Missing sampled-graph training row IDs. "
                    "Run code/scripts/run_sampled_graph_features.py first."
                )
            sampled_row_ids = pd.read_parquet(sampled_path)["_row_id"].to_numpy()
            fit_train &= np.isin(row_ids, sampled_row_ids)
        else:
            fit_train = fraud_preserving_subsample(
                y,
                train,
                float(config["models"].get("full_train_legitimate_fraction", 1.0)),
                int(config["seed"]),
            )
        features = _load_features(group)
        selected = list(features.columns)
        group_x = features.to_numpy(dtype=np.float32)
        del features
        classical = _make_classical_models(config, y[fit_train])
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
                model.fit(
                    group_x[fit_train],
                    y[fit_train],
                    group_x[validation],
                    y[validation],
                )
                validation_scores = model.predict_proba(group_x[validation])
                test_scores = model.predict_proba(group_x[test])
            else:
                model.fit(group_x[fit_train], y[fit_train])
                validation_scores = model.predict_proba(group_x[validation])[:, 1]
                test_scores = model.predict_proba(group_x[test])[:, 1]
            threshold = select_threshold(y[validation], validation_scores)
            metrics = evaluate_predictions(y[test], test_scores, threshold)
            results.append(
                {
                    "model": model_name,
                    "features": group,
                    "training_scope": (
                        "sampled" if group in SAMPLED_FEATURE_GROUPS else "full"
                    ),
                    **metrics,
                    "train_rows": int(fit_train.sum()),
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
                importance_dir = Path("evaluations/metrics")
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
                    Path("evaluations/figures")
                    / f"feature_importance_{model_slug}_{group}.png",
                    dpi=160,
                )
                plt.close()

    results_frame = pd.DataFrame(results)
    results_path = output_dir / "main_results.csv"
    if results_path.is_file():
        previous_results = pd.read_csv(results_path)
        if {"model", "features"}.issubset(previous_results.columns):
            results_frame = pd.concat(
                [previous_results, results_frame], ignore_index=True
            ).drop_duplicates(["model", "features"], keep="last")
    results_frame = results_frame.sort_values(["model", "features"]).reset_index(
        drop=True
    )
    results_frame.to_csv(results_path, index=False)
    _save_plots(results_frame, curves)
    _publish_readme_results(results_frame)
    _save_research_summary(results_frame)
    print(
        results_frame[["model", "features", "pr_auc", "precision", "recall", "f1"]].rename(
            columns={
                "precision": "fraud_precision",
                "recall": "fraud_recall",
                "f1": "fraud_f1",
            }
        )
    )
    return results_frame
