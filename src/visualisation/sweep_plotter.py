# src/visualisation/sweep_plotter.py

import math
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils import save_figure

def plot_sweep_lines(
    df, x_col, y_col,  group_col="perturbation", outdir=None,
    order=None, colour_map=None, title=None, x_label=None, y_label=None, 
):
    """General grouped line plot with markers (for 1D sweeps)."""

    df = df.copy()

    if order is not None:
        df = df[df[group_col].isin(order)]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    groups = order if order is not None else df[group_col].unique()

    for group in groups:
        sub = df[df[group_col] == group].sort_values(x_col)

        if sub.empty:
            continue

        colour = colour_map.get(group) if colour_map else None

        ax.plot(sub[x_col], sub[y_col],
            marker="o", markersize=5, linewidth=2,
            label=group, color=colour,
        )
    ax.margins(x=0)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=7)

    plt.tight_layout()

    save_figure(fig, outdir, title=title)

    plt.show()