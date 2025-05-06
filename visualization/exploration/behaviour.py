# visuals/exploration/behavior.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from matplotlib.ticker import FuncFormatter, EngFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def set_y_axis_format(ax, mode: str = "full"):
    if mode == "short":
        ax.yaxis.set_major_formatter(EngFormatter(sep=""))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

def _filter_nonzero_columns(df, columns, threshold=0.1):
    valid, skipped = [], []
    for col in columns:
        if col not in df.columns:
            skipped.append(col)
            continue
        non_zero_ratio = (df[col] != 0).sum() / len(df[col])
        if non_zero_ratio >= threshold:
            valid.append(col)
        else:
            skipped.append(col)
    return valid, skipped

def plot_raw_series(df: pd.DataFrame, customer_id: str, columns: list = None, non_zero_threshold=0.1, group_by="PodID", format="full"):
    subset = df[df["CustomerID"] == customer_id].copy()

    if columns is None:
        columns = [col for col in df.columns if "Consumption" in col]

    valid_cols, skipped_cols = _filter_nonzero_columns(subset, columns, non_zero_threshold)
    if not valid_cols:
        logger.warning(f"No valid time series to plot for Customer {customer_id}. Skipped: {skipped_cols}")
        return

    n_cols = min(len(valid_cols), 2)
    n_rows = (len(valid_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        grouped = subset.groupby(group_by)

        palette = sns.color_palette("tab10", n_colors=len(grouped))
        for j, (group_id, group_data) in enumerate(grouped):
            ax.plot(group_data.index, group_data[col], label=f"{group_by} {group_id}", color=palette[j])

        ax.set_title(f"{col} - Customer {customer_id}", fontsize=10)
        ax.set_ylabel("kWh")
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True)
        ax.legend(fontsize=8)
        set_y_axis_format(ax, format)

    for j in range(len(valid_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    plt.show()

    if skipped_cols:
        logger.info(f"Skipped (mostly zero or missing): {skipped_cols}")

def plot_overlay_years(df: pd.DataFrame, customer_id: str, columns: list, kind: str = "line", non_zero_threshold=0.1, format="full"):
    assert kind in {"line", "box", "violin"}, "kind must be one of: line, box, violin"
    data = df[df["CustomerID"] == customer_id].copy().reset_index()
    data["Year"] = data["ReportingMonth"].dt.year
    data["Month"] = data["ReportingMonth"].dt.month

    valid_cols, skipped_cols = _filter_nonzero_columns(data, columns, non_zero_threshold)
    if not valid_cols:
        logger.warning(f"No valid columns to plot overlay for customer {customer_id}")
        return

    for col in valid_cols:
        if kind == "line":
            grouped = data.groupby(["Month", "Year"])[col].mean().reset_index()
            pivot = grouped.pivot(index="Month", columns="Year", values=col)
            pivot.plot(title=f"Yearly Overlay - {col} - Customer {customer_id}", figsize=(10, 5), marker='o')
            plt.xticks(ticks=range(1, 13), rotation=90)
            set_y_axis_format(plt.gca(), format)
        elif kind == "box":
            sns.boxplot(data=data, x="Month", y=col)
            plt.title(f"Boxplot - {col} - Customer {customer_id}")
            plt.xticks(rotation=90)
            set_y_axis_format(plt.gca(), format)
        elif kind == "violin":
            sns.violinplot(data=data, x="Month", y=col, inner="quartile", cut=0)
            plt.title(f"Violin Plot - {col} - Customer {customer_id}")
            plt.xticks(rotation=90)
            set_y_axis_format(plt.gca(), format)

        plt.xlabel("Month")
        plt.ylabel("kWh")
        plt.tight_layout()
        plt.show()

    if skipped_cols:
        logger.info(f"Skipped overlay for low-signal columns: {skipped_cols}")

def plot_missingness_heatmap(df: pd.DataFrame, value_column="PeakConsumption"):
    if df.empty or value_column not in df.columns:
        logger.warning("Empty DataFrame or invalid value column for missingness heatmap.")
        return

    temp = df.reset_index()
    pivot = temp.pivot_table(index="PodID", columns="ReportingMonth", values=value_column, aggfunc="size")
    sns.heatmap(pivot.isna(), cbar=False)
    plt.title(f"Missing Data Heatmap: {value_column}")
    plt.xlabel("Month")
    plt.ylabel("PodID")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_top_consumers(df: pd.DataFrame, columns: list, top_n=5, non_zero_threshold=0.1, format="full"):
    valid_cols, skipped_cols = _filter_nonzero_columns(df, columns, non_zero_threshold)
    if not valid_cols:
        logger.warning("No valid consumption types found for top consumer plot.")
        return

    for col in valid_cols:
        totals = df.groupby("CustomerID")[col].sum().nlargest(top_n)
        ax = totals.plot(kind="bar", title=f"Top {top_n} Consumers by {col}")
        set_y_axis_format(ax, format)
        plt.ylabel("Total Consumption (kWh)")
        plt.xlabel("Customer ID")
        plt.tight_layout()
        plt.show()

    if skipped_cols:
        logger.info(f"Skipped low-signal top consumer columns: {skipped_cols}")

def facet_consumption_profiles(df: pd.DataFrame, customer_id: str, columns: list = None, non_zero_threshold=0.1, format="full"):
    sample = df[df["CustomerID"] == customer_id].copy().reset_index()
    if columns is None:
        columns = [col for col in sample.columns if "Consumption" in col]

    valid_cols, skipped_cols = _filter_nonzero_columns(sample, columns, non_zero_threshold)
    if not valid_cols:
        logger.warning(f"No valid data for faceted plot. Skipped: {skipped_cols}")
        return

    melted = sample.melt(id_vars=["ReportingMonth"], value_vars=valid_cols,
                         var_name="ConsumptionType", value_name="kWh")

    g = sns.FacetGrid(melted, col="ConsumptionType", col_wrap=3, height=3.5, sharey=False)
    g.map_dataframe(sns.lineplot, x="ReportingMonth", y="kWh")
    g.set_titles("{col_name}")
    g.set_axis_labels("Date", "kWh")
    for ax in g.axes.flatten():
        set_y_axis_format(ax, format)
        ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

    if skipped_cols:
        logger.info(f"Skipped faceting for low-signal columns: {skipped_cols}")
