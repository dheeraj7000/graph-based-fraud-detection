"""Interactive presentation demo. Run from the repository root:

    streamlit run code/demo/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "evaluations" / "metrics"
PREDICTION_DIR = ROOT / "evaluations" / "predictions"
PROCESSED_DIR = ROOT / "data" / "processed"
DEMO_DIR = PROCESSED_DIR / "demo"

MODELS = ("Logistic", "Random Forest", "XGBoost", "MLP")
FULL_TRACK = ("tabular", "full_graph", "temporal_only", "full_temporal")
SAMPLED_TRACK = ("tabular_sampled", "sampled_graph", "sampled_temporal")
FEATURE_LABELS = {
    "tabular": "Tabular",
    "full_graph": "Tabular + full graph",
    "temporal_only": "Tabular + 24h activity",
    "full_temporal": "Tabular + full graph + 24h",
    "tabular_sampled": "Tabular (sampled rows)",
    "sampled_graph": "Tabular + sampled graph",
    "sampled_temporal": "Tabular + sampled graph + 24h",
}
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "precision": "Fraud precision",
    "recall": "Fraud recall",
    "f1": "Fraud F1",
    "roc_auc": "ROC-AUC",
}
COMPARISON_LABELS = {
    "sampled_graph_gain": "Sampled graph vs tabular (sampled rows)",
    "full_graph_gain": "Full graph vs tabular",
    "full_graph_gain_over_temporal": "Full graph + 24h vs 24h only",
}
GRAPH_METRICS = (
    "pagerank",
    "in_degree",
    "out_degree",
    "weighted_degree",
    "unique_counterparty",
    "community_id",
    "unseen",
)

st.set_page_config(page_title="Graph-Based Fraud Detection Demo", layout="wide")
st.markdown(
    "<style>.block-container {padding-top: 2rem;} "
    "[data-testid='stSidebarUserContent'] {padding-top: 1rem;}</style>",
    unsafe_allow_html=True,
)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / "main_results.csv")


@st.cache_data
def load_gains() -> pd.DataFrame:
    path = METRICS_DIR / "fraud_class_gains.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data
def load_predictions(model: str, features: str) -> pd.DataFrame:
    return pd.read_parquet(PREDICTION_DIR / f"{model}_{features}.parquet")


@st.cache_data
def load_demo_cache() -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    transactions = pd.read_parquet(DEMO_DIR / "demo_transactions.parquet")
    thresholds = pd.read_parquet(DEMO_DIR / "thresholds.parquet")
    lookup = {
        f"{row.model}::{row.features}": float(row.threshold)
        for row in thresholds.itertuples()
    }
    edges = pd.read_parquet(DEMO_DIR / "ego_edges.parquet")
    touches = pd.concat([edges["nameOrig"], edges["nameDest"]]).value_counts()
    transactions["history_edges"] = (
        transactions["nameOrig"].map(touches).fillna(0)
        + transactions["nameDest"].map(touches).fillna(0)
    ).astype(int)
    return transactions, lookup, edges


def available_groups(results: pd.DataFrame, model: str, track: tuple[str, ...]) -> list[str]:
    present = set(results.loc[results["model"] == model, "features"])
    return [group for group in track if group in present]


def section_problem(results: pd.DataFrame) -> None:
    eda = load_json(METRICS_DIR / "eda_summary.json")
    split = load_json(PROCESSED_DIR / "split.json")
    st.header("1 · The imbalance problem")
    columns = st.columns(3)
    columns[0].metric("Transactions", f"{int(eda.get('transactions', 0)) / 1e6:.2f}M")
    columns[1].metric("Fraud transactions", f"{int(eda.get('fraud_count', 0)):,}")
    columns[2].metric("Fraud rate", f"{float(eda.get('fraud_percentage', 0)):.3f}%")
    st.caption(
        "Chronological split by time step: "
        f"{int(split.get('train_steps', 0)):,} train (steps ≤ {split.get('train_end_step')}), "
        f"{int(split.get('validation_steps', 0)):,} validation "
        f"(≤ {split.get('validation_end_step')}), "
        f"{int(split.get('test_steps', 0)):,} test (≤ {split.get('test_end_step')})."
    )

    st.subheader("The accuracy trap")
    left, right = st.columns(2)
    model = left.selectbox("Model", MODELS, index=MODELS.index("XGBoost"), key="trap_model")
    groups = available_groups(results, model, FULL_TRACK + SAMPLED_TRACK)
    features = right.selectbox(
        "Feature set",
        groups,
        index=groups.index("full_graph") if "full_graph" in groups else 0,
        format_func=FEATURE_LABELS.get,
        key="trap_features",
    )
    predictions = load_predictions(model, features)
    y_true = predictions["y_true"].to_numpy()
    overall_rate = float(eda.get("fraud_percentage", 0)) / 100
    test_rate = float(y_true.mean())
    st.write(
        f"About 1 in {1 / overall_rate:,.0f} transactions is fraud overall "
        f"(1 in {1 / test_rate:,.0f} in the test period). A model that never flags anything "
        "still looks excellent on accuracy and on all-class averages. That is why every "
        "precision, recall, and F1 in this project is computed **for the fraud class only**."
    )
    rows = []
    for name, predicted in (
        ("Always predict 'legitimate'", np.zeros_like(y_true)),
        (f"{model} · {FEATURE_LABELS[features]}", predictions["prediction"].to_numpy(int)),
    ):
        rows.append(
            {
                "Classifier": name,
                "Accuracy (all rows)": accuracy_score(y_true, predicted),
                "Weighted F1 (both classes)": f1_score(
                    y_true, predicted, average="weighted", zero_division=0
                ),
                "Fraud precision": precision_score(y_true, predicted, zero_division=0),
                "Fraud recall": recall_score(y_true, predicted, zero_division=0),
                "Fraud F1": f1_score(y_true, predicted, zero_division=0),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(precision=4), hide_index=True, width="stretch"
    )
    st.caption(
        f"Test period: {len(y_true):,} transactions, {int(y_true.sum()):,} fraud. "
        "The trivial classifier scores near-perfect accuracy and weighted F1 while catching "
        "zero fraud; only the fraud-class columns expose the difference."
    )


def section_results(results: pd.DataFrame) -> None:
    st.header("2 · Measured test results (fraud class)")
    sampled = load_json(PROCESSED_DIR / "sampled_graph_metadata.json")
    sample_label = f"Sampled graph ({int(sampled.get('sampled_training_edges', 0)):,} edges)"
    tracks = {"Full graph (every training edge)": FULL_TRACK, sample_label: SAMPLED_TRACK}
    left, right = st.columns([2, 1])
    track_name = left.radio(
        "Experiment track",
        list(tracks),
        index=1 if st.query_params.get("track") == "sampled" else 0,
        horizontal=True,
    )
    metric = right.selectbox(
        "Metric", list(METRIC_LABELS), index=3, format_func=METRIC_LABELS.get
    )
    track = tracks[track_name]
    selected = results.loc[results["features"].isin(track)].copy()
    if selected.empty:
        st.warning("This track has not been run yet.")
        return
    selected["Feature set"] = selected["features"].map(FEATURE_LABELS)
    order = [FEATURE_LABELS[group] for group in track]
    chart = (
        alt.Chart(selected)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, sort=list(MODELS), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Feature set:N", sort=order),
            y=alt.Y(f"{metric}:Q", title=METRIC_LABELS[metric], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Feature set:N", sort=order, legend=alt.Legend(orient="bottom", labelLimit=0, columns=4)),
            tooltip=["model", "Feature set", alt.Tooltip(f"{metric}:Q", format=".4f")],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, width="stretch")
    table = selected.pivot(index="model", columns="Feature set", values=metric)
    st.dataframe(
        table.reindex(index=list(MODELS), columns=order).style.format("{:.4f}"),
        width="stretch",
    )
    with st.expander("All metrics and fraud-class confusion counts"):
        columns = [
            "model",
            "Feature set",
            *METRIC_LABELS,
            "fraud_support",
            "true_positives",
            "false_positives",
            "false_negatives",
            "threshold",
            "train_rows",
        ]
        st.dataframe(
            selected[[column for column in columns if column in selected]]
            .rename(columns=METRIC_LABELS)
            .style.format(precision=4),
            hide_index=True,
            width="stretch",
        )


def section_full_graph(results: pd.DataFrame) -> None:
    st.header("3 · Do full-network structural features move the needle?")
    sampled = load_json(PROCESSED_DIR / "sampled_graph_metadata.json")
    full = load_json(PROCESSED_DIR / "full_graph_metadata.json")
    graph_sizes = pd.DataFrame(
        {
            "Sampled graph": [
                sampled.get(f"sampled_{key}") for key in ("training_edges", "training_accounts",
                                                          "training_frauds", "weak_components")
            ],
            "Full graph": [
                full.get(f"full_{key}") for key in ("training_edges", "training_accounts",
                                                    "training_frauds", "weak_components")
            ],
        },
        index=["Training edges", "Accounts", "Fraud edges", "Weakly connected components"],
    )
    st.dataframe(graph_sizes.style.format("{:,.0f}", na_rep="—"), width="stretch")

    gains = load_gains()
    if gains.empty or not gains["comparison"].isin(["full_graph_gain"]).any():
        st.warning("Run the full-graph experiments to populate this section.")
        return
    gains = gains.loc[gains["comparison"].isin(COMPARISON_LABELS)].copy()
    gains["Comparison"] = gains["comparison"].map(COMPARISON_LABELS)
    metric = st.selectbox(
        "Change in",
        ["f1_gain", "recall_gain", "precision_gain", "pr_auc_gain"],
        format_func=lambda name: METRIC_LABELS[name.removesuffix("_gain")],
    )
    order = [COMPARISON_LABELS[name] for name in COMPARISON_LABELS]
    chart = (
        alt.Chart(gains)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None, sort=list(MODELS), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Comparison:N", sort=order),
            y=alt.Y(f"{metric}:Q", title=f"Δ {METRIC_LABELS[metric.removesuffix('_gain')]}"),
            color=alt.Color("Comparison:N", sort=order, legend=alt.Legend(orient="bottom", labelLimit=0, columns=4)),
            tooltip=["model", "Comparison", alt.Tooltip(f"{metric}:Q", format="+.4f")],
        )
        .properties(height=360)
    )
    rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    st.altair_chart(chart + rule, width="stretch")

    summary = gains.groupby("Comparison", sort=False).agg(
        models_improved=(metric, lambda values: f"{int((values > 0).sum())}/{len(values)}"),
        mean_change=(metric, "mean"),
    )
    st.dataframe(
        summary.reindex([label for label in order if label in summary.index]).style.format(
            {"mean_change": "{:+.4f}"}
        ),
        width="stretch",
    )
    st.caption(
        "Each bar is treatment minus its own baseline for the same model and training rows. "
        "Sampled and full tracks also differ in training-set size, so compare gains, not raw scores."
    )


def draw_ego_network(edges: pd.DataFrame, source: str, destination: str) -> plt.Figure:
    focus = {source, destination}
    direct = edges.loc[edges["nameOrig"].isin(focus) | edges["nameDest"].isin(focus)]
    direct = direct.nlargest(60, "amount")
    neighbors = (set(direct["nameOrig"]) | set(direct["nameDest"])) - focus
    indirect = edges.loc[
        ~edges.index.isin(direct.index)
        & (edges["nameOrig"].isin(neighbors) | edges["nameDest"].isin(neighbors))
    ].nlargest(90, "amount")
    shown = pd.concat([direct, indirect])

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(focus)
    for edge in shown.itertuples():
        graph.add_edge(edge.nameOrig, edge.nameDest, fraud=int(edge.isFraud))
    graph.add_edge(source, destination, fraud=-1)

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    layout = nx.spring_layout(nx.Graph(graph), seed=7, k=0.9)
    colors = [
        "#3071be" if node == source else "#efb145" if node == destination else "#b8c2cf"
        for node in graph.nodes
    ]
    sizes = [420 if node in focus else 90 for node in graph.nodes]
    nx.draw_networkx_nodes(graph, layout, node_color=colors, node_size=sizes, ax=axis)
    for kind, color, style, width in (
        (0, "#9aa5b1", "solid", 0.8),
        (1, "#d62728", "solid", 2.0),
        (-1, "#111f36", "dashed", 2.5),
    ):
        selected = [(u, v) for u, v, data in graph.edges(data=True) if data["fraud"] == kind]
        if selected:
            nx.draw_networkx_edges(
                graph, layout, edgelist=selected, edge_color=color, style=style,
                width=width, arrows=True, arrowsize=10, ax=axis,
            )
    nx.draw_networkx_labels(
        graph, layout, labels={node: node for node in focus}, font_size=8, ax=axis
    )
    axis.set_axis_off()
    axis.set_title(
        "Blue = origin, gold = destination, dashed = this test transaction,\n"
        "red = labelled fraud edges seen in training, grey = other training edges",
        fontsize=9,
    )
    figure.tight_layout()
    return figure


def section_explorer(results: pd.DataFrame) -> None:
    st.header("4 · Transaction explorer (held-out test period)")
    if not (DEMO_DIR / "demo_transactions.parquet").is_file():
        st.warning("Run `python code/scripts/prepare_demo.py` to build the explorer cache.")
        return
    transactions, thresholds, edges = load_demo_cache()
    columns = st.columns(3)
    model = columns[0].selectbox("Model", MODELS, index=MODELS.index("XGBoost"))
    groups = [
        group
        for group in available_groups(results, model, FULL_TRACK + SAMPLED_TRACK)
        if f"p::{model}::{group}" in transactions
    ]
    baseline = columns[1].selectbox(
        "Baseline features", groups, index=0, format_func=FEATURE_LABELS.get
    )
    default_graph = next(
        (groups.index(group) for group in ("full_graph", "sampled_graph") if group in groups), 0
    )
    treatment = columns[2].selectbox(
        "Graph features", groups, index=default_graph, format_func=FEATURE_LABELS.get
    )

    fraud = transactions["isFraud"].to_numpy() == 1
    base_flag = (
        transactions[f"p::{model}::{baseline}"].to_numpy()
        >= thresholds[f"{model}::{baseline}"]
    )
    graph_flag = (
        transactions[f"p::{model}::{treatment}"].to_numpy()
        >= thresholds[f"{model}::{treatment}"]
    )
    category_names = [
        "Fraud caught only with graph features",
        "Fraud caught by both",
        "Fraud caught only by baseline",
        "Fraud missed by both",
        "Legitimate but flagged (false alarm)",
        "Legitimate, correctly cleared (sample)",
    ]
    categories = np.select(
        [
            fraud & graph_flag & ~base_flag,
            fraud & graph_flag & base_flag,
            fraud & ~graph_flag & base_flag,
            fraud & ~graph_flag & ~base_flag,
            ~fraud & (graph_flag | base_flag),
        ],
        category_names[:-1],
        default=category_names[-1],
    )
    counts = pd.Series(categories).value_counts()
    category = st.radio(
        "Show",
        [name for name in category_names if name in counts.index],
        format_func=lambda name: f"{name} ({counts[name]:,})",
        horizontal=True,
    )
    order = st.radio(
        "Sort by",
        ["Largest training-graph neighbourhood", "Highest graph-model fraud probability"],
        horizontal=True,
    )
    sort_column = (
        "history_edges" if order.startswith("Largest") else f"p::{model}::{treatment}"
    )
    subset = transactions.loc[categories == category].sort_values(
        sort_column, ascending=False
    )
    row_label = {
        index: (
            f"row {row._row_id:,} · step {row.step} · {row.type} · "
            f"{row.amount:,.2f} · {row.nameOrig} → {row.nameDest} · "
            f"{row.history_edges} prior edges"
        )
        for index, row in subset.head(300).iterrows()
    }
    chosen = st.selectbox("Transaction", list(row_label), format_func=row_label.get)
    if chosen is None:
        return
    record = transactions.loc[chosen]

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Transaction")
        st.markdown(
            f"**{record['type']}** of **{record['amount']:,.2f}** at step {record['step']}  \n"
            f"`{record['nameOrig']}` → `{record['nameDest']}`  \n"
            f"Origin balance {record['oldbalanceOrg']:,.2f} → {record['newbalanceOrig']:,.2f}  \n"
            f"Destination balance {record['oldbalanceDest']:,.2f} → "
            f"{record['newbalanceDest']:,.2f}  \n"
            f"Ground truth: **{'FRAUD' if record['isFraud'] == 1 else 'legitimate'}**"
        )
        scores = pd.DataFrame(
            [
                {
                    "Feature set": FEATURE_LABELS[group],
                    "Fraud probability": float(record[f"p::{model}::{group}"]),
                    "Threshold": thresholds[f"{model}::{group}"],
                }
                for group in groups
            ]
        )
        scores["Decision"] = np.where(
            scores["Fraud probability"] >= scores["Threshold"], "FLAG", "clear"
        )
        st.subheader(f"{model} scores")
        st.dataframe(
            scores.style.format({"Fraud probability": "{:.4f}", "Threshold": "{:.4f}"}),
            hide_index=True,
            width="stretch",
        )
        if "g::source_pagerank" in transactions:
            st.subheader("Full-graph features")
            graph_table = pd.DataFrame(
                {
                    "Origin": [record[f"g::source_{name}"] for name in GRAPH_METRICS],
                    "Destination": [
                        record[f"g::destination_{name}"] for name in GRAPH_METRICS
                    ],
                },
                index=GRAPH_METRICS,
            )
            st.dataframe(graph_table.style.format("{:.6g}"), width="stretch")
            st.caption(
                "PageRank and component ID are shown for analysis only; models never use "
                "them, because training rows cannot receive them without future edges."
            )
    with right:
        st.subheader("Neighbourhood in the training graph")
        st.pyplot(draw_ego_network(edges, record["nameOrig"], record["nameDest"]))
        history = int(record["history_edges"])
        if history == 0:
            st.info(
                "Neither account appears in the training graph, so the unseen-account flags "
                "are the graph signal here. Many PaySim frauds involve brand-new accounts."
            )
        st.caption(
            f"{history} training-period edges touch these two accounts (up to 60 direct and "
            "90 second-hop edges are drawn, largest amounts first). The model never sees "
            "edges from the validation or test periods."
        )


def main() -> None:
    st.sidebar.title("Graph-Based Fraud Detection")
    st.sidebar.caption("PaySim · CSE 573 · live demo")
    if not (METRICS_DIR / "main_results.csv").is_file():
        st.error("No results found. Run `python code/run_pipeline.py` first.")
        return
    results = load_results()
    sections = {
        "1 · The imbalance problem": section_problem,
        "2 · Results (fraud class)": section_results,
        "3 · Full graph vs sample": section_full_graph,
        "4 · Transaction explorer": section_explorer,
    }
    requested = st.query_params.get("section", "1")
    start = int(requested) - 1 if requested.isdigit() and 1 <= int(requested) <= 4 else 0
    choice = st.sidebar.radio("Section", list(sections), index=start)
    st.sidebar.divider()
    st.sidebar.markdown(
        "Precision, recall, and F1 are **fraud-class only** (`isFraud = 1`). "
        "All numbers come from the pipeline's cached test-set outputs."
    )
    sections[choice](results)


main()
