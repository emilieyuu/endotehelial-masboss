# abm/plotting/timeseries_plotter.py
#
# Plot time-series outputs from ABM result CSVs / DataFrames.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PERTURBATION_ORDER = [
    "WT",
    "DSP_KO",
    "TJP1_KO",
    "JCAD_KO",
    "DSP_JCAD_DKO",
    "TJP1_JCAD_DKO",
]

PERTURBATION_COLOURS = {
    "WT": "black",
    "DSP_KO": "tab:blue",
    "TJP1_KO": "tab:red",
    "JCAD_KO": "tab:green",
    "DSP_JCAD_DKO": "tab:purple",
    "TJP1_JCAD_DKO": "tab:orange",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _prepare_plot_df(df, y_col):
    """Return copy of df with perbs as ordered categorical."""
    df = df.copy()
    df["perturbation"] = pd.Categorical(
        df["perturbation"], categories=PERTURBATION_ORDER, ordered=True,
    )
    df = df.sort_values(["perturbation", "time"])

    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in DataFrame.")

    return df

# ------------------------------------------------------------------
# Generic line plot
# ------------------------------------------------------------------
def plot_metric_timeseries(df, y_col, y_label, 
                           title=None, outdir=None, ylim=None, ax=None
    ):
    """
    Plot one time-series metric with one line per perturbation.

    df: DataFrame with columns ['time', 'perturbation', y_col]
    y_col: metric column to plot
    y_label: y-axis label
    ylim: optional y-axis limits tuple
    ax: optional matplotlib axis
    """
    df = _prepare_plot_df(df, y_col)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        created_fig = True

    for perturbation in PERTURBATION_ORDER:
        sub = df[df["perturbation"] == perturbation]
        if sub.empty:
            continue

        ax.plot(sub["time"], sub[y_col], label=perturbation,
            color=PERTURBATION_COLOURS[perturbation], linewidth=2.0,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label)

    if title is not None: ax.set_title(title)
    if ylim is not None: ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    if created_fig:
        plt.tight_layout()

        if outdir is not None:
            outpath = Path(outdir / f"{y_col}_over_time.png")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")

        plt.show()

# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------
def plot_ar_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="ar",
        y_label="Aspect ratio",
        title="Aspect ratio over time",
        outdir=outdir,
        ax=ax,
    )


def plot_rho_balance_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="rho_balance",
        y_label="Δ = RhoC - RhoA",
        title="Rho balance over time",
        outdir=outdir,
        ax=ax,
    )


def plot_squeeze_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="sf_squeeze",
        y_label="SF squeeze force",
        title="Stress fibre squeeze over time",
        outdir=outdir,
        ax=ax,
    )


def plot_k_ratio_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="cortex_k_ratio",
        y_label=r"$k_{polar} / k_{lateral}$",
        title="Polar/lateral spring stiffness over time",
        outdir=outdir,
        ax=ax,
    )