# src/visualisation/line_plotter.py

import math
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils import save_figure

# ------------------------------------------------------------------
# Single 
# ------------------------------------------------------------------
def plot_metric_timeseries(
    df, y_col, y_label,
    order=None, colour_map=None, title=None, outdir=None,
    ax=None,
):
    """
    Simple time-series plot: one line per perturbation.
    Assumes columns: ['time', 'perturbation', y_col]
    """

    if order is not None:
        df = df[df["perturbation"].isin(order)]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True

    groups = order if order is not None else df["perturbation"].unique()

    for group in groups:
        sub = df[df["perturbation"] == group]
        if sub.empty:
            continue
        colour = colour_map.get(group) if colour_map else None

        ax.plot(sub["time"], sub[y_col], label=group, color=colour)

    ax.margins(x=0)
    ax.set_xlabel("time (min)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.2)

    if created_fig:
        plt.tight_layout()

        if outdir is not None and title is not None:
            save_figure(fig, outdir, title=title)


        plt.show()


# ------------------------------------------------------------------
# Multi
# ------------------------------------------------------------------
def plot_timeseries_grid(
    df, metrics, ncols=2, title=None,
    order=None, colour_map=None, outdir=None,
):
    """
    Plot multiple time-series in a grid.

    metrics = [
        {"y_col": "ar", "y_label": "Aspect ratio", "title": "AR"},
        ...
    ]
    """

    nplots = len(metrics)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # --- plot each metric ---
    for ax, metric in zip(axes, metrics):
        plot_metric_timeseries(
            df=df,
            y_col=metric["y_col"],
            y_label=metric["y_label"],
            title=metric.get("title"),
            ax=ax,
            order=order,
            colour_map=colour_map,
        )
        ax.legend().remove() # remove legend

    if title:
        fig.suptitle(title, fontsize=20 , y=0.92)

    # --- shared legend ---
    if order is not None: 
        labels = order
    else: 
        labels = df["perturbation"].unique()

    handles = [
        plt.Line2D([0], [0], color=colour_map.get(lbl, "black"), lw=2)
        for lbl in labels
    ]

    fig.legend(handles, labels, loc="center left", 
               bbox_to_anchor=(1, 0.5), ncol=1, fontsize=7, frameon=False,)

    # hide unused axes
    for ax in axes[nplots:]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if outdir is not None and title is not None:
        save_figure(fig, outdir, title=title)

    plt.show()