from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_eda(frame: pd.DataFrame, figure_dir: str | Path) -> dict[str, object]:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fraud = frame["isFraud"].astype(bool)
    report = {
        "transactions": int(len(frame)),
        "fraud_count": int(fraud.sum()),
        "fraud_percentage": float(fraud.mean() * 100),
        "fraud_by_type": frame.groupby("type", observed=True)["isFraud"]
        .agg(["sum", "count", "mean"])
        .to_dict(orient="index"),
        "unique_source_accounts": int(frame["nameOrig"].nunique()),
        "unique_destination_accounts": int(frame["nameDest"].nunique()),
        "source_reuse_frequency": frame["nameOrig"].value_counts().describe().to_dict(),
        "destination_reuse_frequency": frame["nameDest"]
        .value_counts()
        .describe()
        .to_dict(),
        "transaction_types": sorted(frame["type"].astype(str).unique().tolist()),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    plots = (
        ("fraud_distribution.png", lambda ax: frame["isFraud"].value_counts().sort_index().plot.bar(ax=ax)),
        ("fraud_by_type.png", lambda ax: frame.groupby("type", observed=True)["isFraud"].sum().plot.bar(ax=ax)),
        ("transaction_count_by_step.png", lambda ax: frame.groupby("step", sort=True).size().plot(ax=ax)),
        ("fraud_count_by_step.png", lambda ax: frame.groupby("step", sort=True)["isFraud"].sum().plot(ax=ax)),
        ("transaction_amount_distribution.png", lambda ax: frame["amount"].clip(upper=frame["amount"].quantile(0.99)).plot.hist(bins=50, ax=ax)),
        ("fraud_amount_distribution.png", lambda ax: frame.loc[fraud, "amount"].plot.hist(bins=50, ax=ax)),
    )
    for name, draw in plots:
        fig, ax = plt.subplots(figsize=(9, 5))
        draw(ax)
        ax.set_title(name.removesuffix(".png").replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(figure_dir / name, dpi=160)
        plt.close(fig)
    return report
