from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_bar_metrics(
    df, x_col, y_cols,
    order=None, colour_map=None, ref_category="WT", 
    title=None, outdir=None, ylabel=None, yerr_cols=None,
):
    """
    General bar plot for one or more metrics across categories.

    df: DataFrame 
    x_col: str – column containing category labels
    y_cols: str or list[str] – one or more numeric columns to plot
    outdir: optional output directory for saving figure
    order: list[str] – explicit category order.
    colour_map: dict — optional mapping from category name -> colour
    ref_category: str - category used for horizontal reference line
    ylabel: str – optional Y-axis label.
    title: str — optional plot title.
    """

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if yerr_cols is not None and isinstance(yerr_cols, str):
        yerr_cols = [yerr_cols]

    if order is not None:
        df = df.set_index(x_col).reindex(order).dropna(how="all").reset_index()

    labels = df[x_col].values
    x = np.arange(len(labels))
    width = 0.8 / len(y_cols)

    fig, ax = plt.subplots(figsize=(7, 4))

    # --- bars ---
    for i, col in enumerate(y_cols):
        values = df[col].values

        # error bars
        if yerr_cols is not None and yerr_cols[i] is not None:
            yerr = df[yerr_cols[i]].values
        else:
            yerr = None

        # colour 
        if colour_map is not None and len(y_cols) == 1:
            colors = [colour_map.get(lbl, "gray") for lbl in labels]
        else:
            colors = None

        ax.bar(
            x + i * width, values, width, 
            label=col, color=colors, edgecolor="black", 
            yerr=yerr, capsize=3,
        )

    # --- reference line  ---
    if ref_category in labels:
        ref_val = df.loc[df[x_col] == ref_category, y_cols[0]].iloc[0]
        ax.axhline(ref_val, linestyle="--", linewidth=1, color="black", alpha=0.6)

    # --- axes ---
    ax.set_xticks(x + width * (len(y_cols) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if len(y_cols) > 1:
        ax.legend()

    plt.tight_layout()

    # --- save ---
    if outdir is not None and title is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        filename = title.lower().replace(" ", "_") + ".png"
        plt.savefig(outdir / filename, dpi=300, bbox_inches="tight")

    plt.show()
