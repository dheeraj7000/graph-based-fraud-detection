from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "evaluations" / "metrics"
RESULTS_PATH = METRICS_DIR / "main_results.csv"
EDA_PATH = METRICS_DIR / "eda_summary.json"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_METADATA_PATH = PROCESSED_DIR / "split_metadata.parquet"
METADATA_PATH = PROCESSED_DIR / "sampled_graph_metadata.json"
FULL_METADATA_PATH = PROCESSED_DIR / "full_graph_metadata.json"
DEMO_SCREENS_DIR = ROOT / "presentations" / "demo_screens"
OUTPUT_PATH = ROOT / "presentations" / "Group19-Fraud-Detection-with-Graph-Databases-FA26-Group-DEMO-Presentation.pptx"

GROUP = "Group 19"
TITLE = "Fraud Detection with Graph Databases"
REPO_URL = "https://github.com/dheeraj7000/graph-based-fraud-detection"
DATASET_URL = "https://www.kaggle.com/datasets/ealaxi/paysim1"
TEAM = (
    ("Vatsal Joshi", "vjoshi29@asu.edu", "1229509166"),
    ("Dheeraj Kumar", "dkumar70@asu.edu", "1238144975"),
    ("Yaksh Pancholi", "ypancho1@asu.edu", "1239803113"),
    ("Lovinder Singh", "Lsingh38@asu.edu", "1238337284"),
    ("Samarth Ogale", "sogale@asu.edu", "1237833014"),
)
PLAN = (
    ("Dataset acquisition, EDA, preprocessing, chronological split", "Yaksh Pancholi", "Oct 24", "Done"),
    ("Causal 24-hour activity features (DuckDB windows)", "Lovinder Singh", "Oct 25", "Done"),
    ("Account-graph features: 50,000-edge sample", "Dheeraj Kumar", "Oct 26", "Done"),
    ("Models (LR, RF, XGBoost, MLP) and threshold tuning", "Lovinder Singh", "Oct 26", "Done"),
    ("Project presentation", "All members", "Oct 26", "Done"),
    ("Fraud-class metrics and ablation study (feedback)", "Vatsal Joshi", "Nov 25", "Done"),
    ("Full training graph (6.08M edges), leakage audit", "Dheeraj Kumar", "Nov 25", "Done"),
    ("Streamlit demo UI and visualizations", "Samarth Ogale", "Nov 25", "Done"),
    ("Multi-seed robustness and cost-sensitive thresholds", "Lovinder Singh, Vatsal Joshi", "Dec 2", "Planned"),
    ("Final report and repository clean-up", "All members", "Dec 6", "Planned"),
)

NAVY = RGBColor(17, 31, 54)
BLUE = RGBColor(48, 113, 190)
TEAL = RGBColor(35, 166, 154)
GOLD = RGBColor(239, 177, 69)
INK = RGBColor(36, 48, 65)
MUTED = RGBColor(102, 116, 135)
PALE = RGBColor(239, 244, 249)
WHITE = RGBColor(255, 255, 255)
SOFT = RGBColor(207, 220, 236)
PALETTE = (BLUE, TEAL, GOLD)
MODELS = ("Logistic", "Random Forest", "XGBoost", "MLP")
SAMPLED_FEATURES = ("tabular_sampled", "sampled_graph", "sampled_temporal")
FULL_FEATURES = ("tabular", "full_graph", "temporal_only", "full_temporal")
FEATURE_LABELS = {
    "tabular_sampled": "Tabular",
    "sampled_graph": "+ graph",
    "sampled_temporal": "+ graph + 24h",
    "tabular": "Tabular",
    "full_graph": "+ graph",
    "temporal_only": "+ 24h",
    "full_temporal": "+ graph + 24h",
}
SCREENS = {
    "accuracy": "01_accuracy_trap.png",
    "results": "02_results.png",
    "full_graph": "03_full_graph.png",
    "explorer": "04_explorer.png",
}
SIDEBAR_FRACTION = 280 / 1280


def _set_background(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_text(
    slide,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 18,
    color: RGBColor = INK,
    bold: bool = False,
    font: str = "Aptos",
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape


def _add_bullets(slide, items: list[str], x: float, y: float, width: float, height: float,
                 *, size: float = 16, color: RGBColor = INK, spacing: float = 8) -> None:
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    frame.margin_right = Inches(0.05)
    frame.margin_top = Inches(0.04)
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = f"•  {item}"
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(spacing)


def _add_footer(slide, page: int, dark: bool = False) -> None:
    color = RGBColor(183, 197, 214) if dark else MUTED
    _add_text(slide, f"CSE 573  •  {GROUP.upper()}  •  {TITLE.upper()}",
              0.55, 7.12, 10.9, 0.2, size=8, color=color)
    _add_text(slide, f"{page:02d}", 12.15, 7.08, 0.55, 0.25, size=9, color=color,
              align=PP_ALIGN.RIGHT)


def _new_slide(prs: Presentation, title: str, page: int, kicker: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, WHITE)
    _add_text(slide, kicker.upper(), 0.6, 0.36, 11.8, 0.28, size=10, color=TEAL, bold=True)
    _add_text(slide, title, 0.6, 0.72, 12.0, 0.62, size=27, color=NAVY, bold=True)
    _add_footer(slide, page)
    return slide


def _add_card(slide, title: str, body: str, x: float, y: float, width: float,
              height: float, accent: RGBColor = BLUE, body_size: float = 14) -> None:
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.08
    card.fill.solid()
    card.fill.fore_color.rgb = PALE
    card.line.fill.background()
    _add_text(slide, title, x + 0.2, y + 0.14, width - 0.4, 0.4,
              size=16, color=accent, bold=True)
    _add_text(slide, body, x + 0.2, y + 0.58, width - 0.4, height - 0.68,
              size=body_size, color=INK)


def _add_table(slide, rows: list[list[str]], x: float, y: float, width: float,
               height: float, *, size: float = 11, left_columns: int = 1,
               column_widths: list[float] | None = None) -> None:
    shape = slide.shapes.add_table(
        len(rows), len(rows[0]), Inches(x), Inches(y), Inches(width), Inches(height)
    )
    table = shape.table
    if column_widths:
        for index, column_width in enumerate(column_widths):
            table.columns[index].width = Inches(column_width)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if row_index == 0 else (
                PALE if row_index % 2 else WHITE
            )
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = (
                PP_ALIGN.LEFT if column_index < left_columns else PP_ALIGN.CENTER
            )
            paragraph.font.name = "Aptos"
            paragraph.font.size = Pt(size)
            paragraph.font.bold = row_index == 0
            paragraph.font.color.rgb = WHITE if row_index == 0 else INK


def _add_screenshot(slide, path: Path, x: float, y: float, max_width: float,
                    max_height: float, caption: str) -> None:
    with Image.open(path) as image:
        aspect = image.width * (1 - SIDEBAR_FRACTION) / image.height
    height = min(max_height, max_width / aspect)
    width = height * aspect
    left = x + (max_width - width) / 2
    picture = slide.shapes.add_picture(
        str(path), Inches(left), Inches(y), Inches(width), Inches(height)
    )
    picture.crop_left = SIDEBAR_FRACTION
    picture.line.color.rgb = RGBColor(208, 217, 228)
    picture.line.width = Pt(1)
    _add_text(slide, caption, x, y + height + 0.1, max_width, 0.6, size=12,
              color=MUTED, align=PP_ALIGN.CENTER)


def _signed(value: float) -> str:
    return f"{value:+.4f}".replace("-", "\u2212")


def _require_inputs() -> None:
    for path, stage in (
        (RESULTS_PATH, "code/scripts/run_experiments.py"),
        (EDA_PATH, "code/scripts/run_preprocessing.py"),
        (SPLIT_METADATA_PATH, "code/scripts/run_preprocessing.py"),
        (METADATA_PATH, "code/scripts/run_sampled_graph_features.py"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run {stage} first.")


def _complete(results: pd.DataFrame, features: tuple[str, ...]) -> bool:
    pairs = set(zip(results["model"], results["features"]))
    return {(model, feature) for model in MODELS for feature in features} <= pairs


def create_deck(output_path: Path = OUTPUT_PATH) -> Path:
    _require_inputs()
    results = pd.read_csv(RESULTS_PATH)
    if not _complete(results, SAMPLED_FEATURES):
        raise ValueError("The deck requires the completed 12-run sampled matrix")
    full_complete = _complete(results, FULL_FEATURES)
    eda = json.loads(EDA_PATH.read_text(encoding="utf-8"))
    sampled = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    full_graph = (
        json.loads(FULL_METADATA_PATH.read_text(encoding="utf-8"))
        if FULL_METADATA_PATH.is_file() else None
    )
    split = pd.read_parquet(SPLIT_METADATA_PATH, columns=["step", "isFraud", "split"])
    split_stats = split.groupby("split").agg(
        rows=("isFraud", "size"), frauds=("isFraud", "sum"),
        first=("step", "min"), last=("step", "max"),
    )
    full_rows = results.loc[results["features"].isin(FULL_FEATURES), "train_rows"]
    full_train_rows = int(full_rows.iloc[0]) if len(full_rows) else int(split_stats.loc["train", "rows"])
    f1 = results.pivot(index="model", columns="features", values="f1").reindex(MODELS)
    indexed = results.set_index(["model", "features"])

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    page = 1

    # Cover: group number, project title, group members
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, NAVY)
    for x, color, size in ((9.5, BLUE, 2.0), (10.45, TEAL, 1.45), (11.2, GOLD, 0.95)):
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x), Inches(0.9 + (x - 9.5) * 0.35), Inches(size), Inches(size)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = color
        circle.line.fill.background()
    _add_text(slide, f"CSE 573  •  FALL 2026  •  {GROUP.upper()}", 0.8, 0.95, 8.0, 0.4,
              size=16, color=TEAL, bold=True)
    _add_text(slide, TITLE, 0.8, 1.5, 8.6, 1.5, size=40, color=WHITE, bold=True)
    _add_text(slide, "Project demo: graph-based fraud detection on 6.36M PaySim transactions",
              0.84, 3.05, 8.6, 0.6, size=19, color=SOFT)
    _add_text(slide, "GROUP MEMBERS", 0.84, 4.3, 8.0, 0.3, size=11, color=TEAL, bold=True)
    for index, (name, email, asu_id) in enumerate(TEAM):
        x = 0.84 + index * 2.4
        _add_text(slide, name, x, 4.67, 2.3, 0.35, size=15, color=WHITE, bold=True)
        _add_text(slide, f"{email}\n{asu_id}", x, 5.05, 2.3, 0.6, size=11, color=SOFT)
    _add_footer(slide, page, dark=True)
    page += 1

    # Problem definition
    slide = _new_slide(prs, "Problem definition", page, "Problem")
    page += 1
    _add_text(
        slide,
        "Given a stream of mobile-money transactions between accounts, flag the fraudulent ones, using both each transaction's own attributes and the structure of the account network it belongs to.",
        0.8, 1.5, 11.7, 1.0, size=19, color=NAVY, bold=True,
    )
    _add_card(slide, "Input → output", "Input: a transaction (type, amount, balances, origin and destination accounts) plus the history of the account graph before it.\nOutput: a fraud probability and a fraud / not-fraud decision.", 0.8, 2.75, 3.75, 2.55, BLUE)
    _add_card(slide, "Why it is hard", f"Only {eda['fraud_percentage']:.3f}% of transactions are fraud, so a model that never flags fraud is {1 - eda['fraud_percentage'] / 100:.1%} accurate. Features must not leak the future, and the graph has millions of accounts.", 4.8, 2.75, 3.75, 2.55, TEAL)
    _add_card(slide, "Research questions", "RQ1  Do account-graph features raise fraud-class F1 over transaction features alone?\nRQ2  Does the full training graph help more than a 50,000-edge sample?\nRQ3  Does 24-hour account activity add fraud-class signal?", 8.8, 2.75, 3.75, 2.55, GOLD, 13)
    _add_text(slide, "Success is measured on the minority fraud class only: fraud precision, recall, F1, and PR-AUC on a held-out future period.",
              0.8, 5.65, 11.7, 0.5, size=15, color=MUTED, align=PP_ALIGN.CENTER)

    # System architecture
    slide = _new_slide(prs, "System architecture", page, "System architecture & algorithms")
    page += 1
    full_edges = f"{int(full_graph['full_training_edges']):,}" if full_graph else "all"
    steps = [
        ("Ingest", "PaySim CSV → typed pandas frame; schema and label checks"),
        ("Split", "Chronological train / validation / test by time step"),
        ("Features", f"Tabular · causal 24h activity · account graph (sampled 50,000 or full {full_edges} edges)"),
        ("Models", "LR, Random Forest, XGBoost, MLP; threshold tuned on validation"),
        ("Serve", "Test predictions cached as Parquet → Streamlit demo app"),
    ]
    for index, (title, body) in enumerate(steps):
        x = 0.65 + index * 2.55
        _add_card(slide, f"{index + 1}  {title}", body, x, 1.65, 2.25, 2.35, PALETTE[index % 3], 13)
        if index < len(steps) - 1:
            _add_text(slide, "→", x + 2.25, 2.55, 0.3, 0.45, size=21, color=MUTED, bold=True, align=PP_ALIGN.CENTER)
    _add_card(slide, "Graph store", "Accounts are nodes and transactions are directed edges weighted by amount. The graph lives in-process as SciPy sparse matrices, with DuckDB for windowed history queries, instead of a separate graph database server: the same graph analytics without a database service.", 0.65, 4.3, 6.05, 2.4, TEAL, 13)
    _add_card(slide, "Technology", "Python 3.13 · pandas · DuckDB · SciPy sparse graphs · scikit-learn · XGBoost · PyTorch · Streamlit + Altair · NetworkX (plots only)\nOne command runs every stage: python code/run_pipeline.py", 6.95, 4.3, 5.75, 2.4, BLUE, 13)

    # Algorithms
    slide = _new_slide(prs, "Algorithms", page, "System architecture & algorithms")
    page += 1
    _add_card(slide, "Graph features (per origin and destination account)", "Model inputs: in / out degree, amount-weighted degree, unique counterparties, unseen-account flag\nAnalysis only: weighted PageRank (sparse power iteration, damping 0.85) and weakly connected components; training rows cannot get them without future edges, so models never use them", 0.65, 1.55, 6.05, 2.5, BLUE, 13)
    _add_card(slide, "Leakage control", "Training rows use a causal prefix: each account's counts include only earlier time steps (DuckDB window: unbounded preceding to 1 preceding).\nValidation and test rows read the frozen training graph.\n24h activity uses the window [step − 24, step), excluding the current step.", 6.95, 1.55, 5.75, 2.5, TEAL, 13)
    _add_card(slide, "Classifiers (all class-weighted for imbalance)", "Logistic Regression: balanced class weights\nRandom Forest: 200 trees, depth 20, balanced subsample\nXGBoost: 400 trees, depth 6, scale_pos_weight = negatives / positives\nMLP: 128-64-32 ReLU, positive-class loss weight, early stopping", 0.65, 4.3, 6.05, 2.4, GOLD, 13)
    _add_card(slide, "Decision threshold", "Each model outputs a fraud probability. The threshold that maximizes fraud-class F1 on the validation period is fixed, then applied once to the test period, so test results are never used for tuning.", 6.95, 4.3, 5.75, 2.4, BLUE, 13)

    # Dataset
    slide = _new_slide(prs, "Dataset: PaySim mobile-money transactions", page, "Datasets")
    page += 1
    _add_card(slide, "Description", "Synthetic transactions generated by the PaySim simulator from real mobile-money logs (Kaggle, CC BY-SA 4.0). Each row is one transaction between two accounts during one hourly step.", 0.65, 1.5, 4.0, 2.35, BLUE, 13)
    size_rows = [
        ["Property", "Value"],
        ["Transactions", f"{eda['transactions']:,}"],
        ["Fraud transactions", f"{eda['fraud_count']:,} ({eda['fraud_percentage']:.3f}%)"],
        ["Time span", f"{split['step'].nunique()} hourly steps (~31 days)"],
        ["Origin / destination accounts", f"{eda['unique_source_accounts']:,} / {eda['unique_destination_accounts']:,}"],
        ["Columns · raw size", "11 · 494 MB CSV"],
        ["Missing values · duplicates", f"{sum(eda['missing_values'].values())} · {eda['duplicate_rows']}"],
    ]
    _add_text(slide, "Size", 0.85, 4.05, 3.0, 0.35, size=16, color=TEAL, bold=True)
    _add_table(slide, size_rows, 0.65, 4.45, 4.0, 2.3, size=10, column_widths=[1.95, 2.05])
    _add_text(slide, "Preprocessing steps", 4.95, 1.5, 7.5, 0.35, size=16, color=TEAL, bold=True)
    fraud_types = {name: value["sum"] for name, value in eda["fraud_by_type"].items() if value["sum"]}
    _add_bullets(
        slide,
        [
            "Load with explicit types; reject missing columns, missing values, or non-binary labels.",
            "Drop account IDs and the isFlaggedFraud rule output from the predictors (IDs only feed the graph and activity features).",
            "Engineer 17 tabular features: amount, balances, balance changes, amount-to-balance ratios, zero-balance flags, one-hot type.",
            "Fraud occurs only in " + " and ".join(f"{name} ({count:,})" for name, count in fraud_types.items()) + ".",
            "Split chronologically by step (70 / 15 / 15% of steps):",
        ],
        4.95, 1.9, 7.6, 2.75, size=13, spacing=4,
    )
    split_rows = [["Partition", "Steps", "Rows", "Fraud"]]
    for name in ("train", "validation", "test"):
        stats = split_stats.loc[name]
        split_rows.append([
            name.capitalize(), f"{int(stats['first'])}–{int(stats['last'])}",
            f"{int(stats['rows']):,}", f"{int(stats['frauds']):,} ({stats['frauds'] / stats['rows']:.2%})",
        ])
    _add_table(slide, split_rows, 5.25, 4.6, 7.2, 1.45, size=11)
    _add_text(slide, f"Model training rows: sampled track {int(sampled['sampled_training_edges']):,} (seed {int(sampled['seed'])}); full-graph track {full_train_rows:,} (all {int(split_stats.loc['train', 'frauds']):,} training frauds + 10% of legitimate rows). Validation and test always stay full-size.",
              4.95, 6.15, 7.6, 0.6, size=12, color=MUTED)

    # Evaluation: metrics and experiments
    slide = _new_slide(prs, "Evaluation: metrics and experiments", page, "Evaluations")
    page += 1
    test_rate = split_stats.loc["test", "frauds"] / split_stats.loc["test", "rows"]
    _add_card(slide, "Metrics (fraud class only)", f"Fraud precision, recall, and F1 at the validation-tuned threshold, with isFraud = 1 as the positive class; no weighted or macro averages.\nPR-AUC (threshold-free) and ROC-AUC as supporting metrics.\nWhy not accuracy: never flagging fraud scores {1 - test_rate:.3f} test accuracy.", 0.65, 1.5, 4.0, 3.55, BLUE, 13)
    _add_card(slide, "Sampled track · 12 runs", f"4 models × 3 feature sets, all trained on the same {int(sampled['sampled_training_edges']):,} sampled training rows.\nTabular\nTabular + sampled graph\nTabular + sampled graph + 24h", 4.9, 1.5, 3.75, 3.55, TEAL, 13)
    full_text = (
        f"4 models × 4 feature sets. Graph from all {full_edges} training edges; models train on every training fraud plus a seeded 10% of legitimate rows ({full_train_rows:,} rows).\nTabular\nTabular + full graph\nTabular + 24h\nTabular + full graph + 24h"
    )
    _add_card(slide, "Full-graph track · 16 runs", full_text, 8.9, 1.5, 3.8, 3.55, GOLD, 13)
    _add_text(slide, "Design", 0.85, 5.3, 2.0, 0.35, size=16, color=TEAL, bold=True)
    _add_bullets(
        slide,
        [
            "Ablation: within each track, feature sets differ by exactly one block, so a change in fraud F1 is attributable to that block.",
            "Same validation and test periods for all 28 runs; feature caches are keyed by row ID and checked for alignment.",
        ],
        0.85, 5.65, 11.9, 1.2, size=13, spacing=4,
    )

    # Evaluation: preliminary findings
    slide = _new_slide(prs, "Evaluation: preliminary findings (test fraud F1)", page, "Evaluations")
    page += 1
    header = ["Model"] + [f"Sampled: {FEATURE_LABELS[f]}" for f in SAMPLED_FEATURES] + [
        f"Full: {FEATURE_LABELS[f]}" for f in FULL_FEATURES
    ]
    rows = [header]
    for model in MODELS:
        row = [model]
        for feature in (*SAMPLED_FEATURES, *FULL_FEATURES):
            value = f1.loc[model].get(feature)
            row.append("—" if pd.isna(value) else f"{value:.4f}")
        rows.append(row)
    _add_table(slide, rows, 0.65, 1.5, 12.05, 2.2, size=11,
               column_widths=[1.45] + [1.514] * 7)
    sampled_graph_delta = f1["sampled_graph"] - f1["tabular_sampled"]
    findings = [
        f"RQ1 (sample): sampled graph features raised fraud F1 for {int((sampled_graph_delta > 0).sum())}/4 models (mean {_signed(sampled_graph_delta.mean())}).",
    ]
    if full_complete:
        full_delta = f1["full_graph"] - f1["tabular"]
        temporal_delta = f1["full_temporal"] - f1["temporal_only"]
        best_model, best_feature = f1[list(FULL_FEATURES)].stack().idxmax()
        best = indexed.loc[(best_model, best_feature)]
        activity_delta = f1["temporal_only"] - f1["tabular"]
        findings += [
            f"RQ2 (full graph): full-graph features raised fraud F1 for {int((full_delta > 0).sum())}/4 models (mean {_signed(full_delta.mean())}); "
            f"on top of 24h activity for {int((temporal_delta > 0).sum())}/4 (mean {_signed(temporal_delta.mean())}).",
            f"So full-network structure does not move the needle: no model's fraud F1 changes by more than {full_delta.abs().max():.4f}, "
            f"and tabular features alone already reach F1 {f1.loc['Random Forest', 'tabular']:.4f} (Random Forest) and {f1.loc['XGBoost', 'tabular']:.4f} (XGBoost).",
            f"RQ3: 24h activity raised full-track fraud F1 for {int((activity_delta > 0).sum())}/4 models (mean {_signed(activity_delta.mean())}). "
            f"Best: {best_model}, {FEATURE_LABELS[best_feature].lstrip('+ ')}: precision {best['precision']:.4f}, recall {best['recall']:.4f}.",
        ]
    else:
        findings.append("RQ2 (full graph): full-track runs still in progress; completed cells are shown, the rest marked —.")
    findings.append("Preliminary: one seed, synthetic data; the two tracks also differ in training-set size.")
    _add_bullets(slide, findings, 0.75, 3.95, 11.9, 2.9, size=13, spacing=6)

    # UI interfaces and data visualization
    screens = {key: DEMO_SCREENS_DIR / name for key, name in SCREENS.items()}
    slide = _new_slide(prs, "UI: interactive demo app (Streamlit)", page, "UI interfaces & data visualization")
    page += 1
    if screens["accuracy"].is_file():
        _add_screenshot(slide, screens["accuracy"], 0.55, 1.5, 6.0, 4.55,
                        "Accuracy trap: a never-fraud classifier looks excellent on accuracy and weighted F1, yet scores zero on the fraud class.")
    if screens["results"].is_file():
        _add_screenshot(slide, screens["results"], 6.8, 1.5, 6.0, 4.55,
                        "Results: fraud-class metric per model and feature set, switchable between the sampled and full-graph tracks.")
    slide = _new_slide(prs, "UI: graph analysis and transaction explorer", page, "UI interfaces & data visualization")
    page += 1
    left = screens["full_graph"] if screens["full_graph"].is_file() else None
    if left is not None:
        _add_screenshot(slide, left, 0.55, 1.5, 6.0, 4.55,
                        "Full graph vs sample: graph size and per-model change in fraud F1, recall, and precision from structural features.")
    if screens["explorer"].is_file():
        _add_screenshot(slide, screens["explorer"], 6.8 if left is not None else 3.65, 1.5, 6.0, 4.55,
                        "Explorer: pick any held-out test transaction to see every model's score, the frozen threshold, and its training-graph neighbourhood.")

    # Project plan
    slide = _new_slide(prs, "Project plan: tasks, deadlines, division of work", page, "Project plan")
    page += 1
    plan_rows = [["Task", "Owner", "Deadline", "Status"]] + [list(row) for row in PLAN]
    _add_table(slide, plan_rows, 0.65, 1.5, 12.05, 4.9, size=12, left_columns=2,
               column_widths=[6.2, 3.15, 1.3, 1.4])
    _add_text(slide, "All members review results and contribute to presentations and the final report.",
              0.65, 6.55, 12.0, 0.4, size=12, color=MUTED)

    # Closing with repository link
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_background(slide, NAVY)
    _add_text(slide, "THANK YOU  •  QUESTIONS", 0.8, 0.9, 8.0, 0.4, size=13, color=TEAL, bold=True)
    _add_text(slide, "Code, data instructions, and evaluations", 0.8, 1.45, 11.5, 0.8, size=30, color=WHITE, bold=True)
    _add_text(slide, "GITHUB REPOSITORY", 0.84, 2.6, 6.0, 0.3, size=11, color=TEAL, bold=True)
    link = _add_text(slide, REPO_URL, 0.84, 2.95, 11.5, 0.55, size=22, color=WHITE, bold=True)
    link.click_action.hyperlink.address = REPO_URL
    for index, (folder, body) in enumerate((
        ("code/", "Pipeline, models, demo app, tests"),
        ("data/", "Download steps and a 20,000-row sample"),
        ("evaluations/", "Metrics, figures, research summary"),
    )):
        x = 0.84 + index * 4.0
        _add_text(slide, folder, x, 3.95, 3.7, 0.4, size=17, color=GOLD, bold=True, font="Consolas")
        _add_text(slide, body, x, 4.38, 3.7, 0.6, size=13, color=SOFT)
    _add_text(slide, "Run the demo:  streamlit run code/demo/app.py", 0.84, 5.35, 11.5, 0.4,
              size=14, color=SOFT, font="Consolas")
    _add_text(slide, f"Dataset: PaySim, Synthetic Financial Datasets for Fraud Detection (Kaggle, CC BY-SA 4.0)  {DATASET_URL}",
              0.84, 6.1, 11.6, 0.5, size=11, color=RGBColor(183, 197, 214))
    _add_footer(slide, page, dark=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    return output_path


def main() -> None:
    output = create_deck(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else OUTPUT_PATH)
    print(f"Saved presentation: {output} ({len(Presentation(output).slides)} slides)")


if __name__ == "__main__":
    main()
