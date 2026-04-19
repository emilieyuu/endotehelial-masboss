# abm/plotting/timeseries_plotter.py
#
# Plot time-series outputs from ABM result CSVs / DataFrames.
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PERTURBATION_COLOURS = {
    "DSP_KO": "tab:blue",
    "DSP_JCAD_DKO": "tab:purple",

    "WT": "tab:gray",
    "JCAD_KO": "tab:green",

    "TJP1_KO": "tab:red",
    "TJP1_JCAD_DKO": "tab:orange",
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _prepare_plot_df(df, y_col):
    """Return copy of df with perturbations ordered by colour dict."""
    df = df.copy()

    order = list(PERTURBATION_COLOURS.keys())

    df = df[df["perturbation"].isin(order)]
    df["perturbation"] = pd.Categorical(
        df["perturbation"], categories=order, ordered=True,
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

    for perturbation, colour in PERTURBATION_COLOURS.items():
        sub = df[df["perturbation"] == perturbation]
        if sub.empty:
            continue

        ax.plot(
            sub["time"], sub[y_col],
            label=perturbation,
            color=colour,
            linewidth=2.0,
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
# Steady-State bar chart
# ------------------------------------------------------------------
def plot_ss_ar_bars(cell_ss_df, outdir=None):
    """
    Steady-state AR bar chart using colour dict for BOTH ordering and colours.
    """

    df = cell_ss_df.copy()

    # --- Enforce ordering from colour dict ---
    order = list(PERTURBATION_COLOURS.keys())
    df = df[df["perturbation"].isin(order)]
    df["perturbation"] = pd.Categorical(df["perturbation"], categories=order, ordered=True)
    df = df.sort_values("perturbation")

    labels = df["perturbation"].values
    values = df["ar"].values
    colors = [PERTURBATION_COLOURS[p] for p in labels]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7.5, 4))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    # --- Value labels ---
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --- WT reference line ---
    if "WT" in df["perturbation"].values:
        wt_ar = df.loc[df["perturbation"] == "WT", "ar"].iloc[0]
        ax.axhline(wt_ar, linestyle="--", linewidth=1.0, color="black", alpha=0.6)

    # --- Formatting ---
    ax.set_ylabel("Aspect ratio (AR)")
    ax.set_title("Steady-state elongation")
    ax.grid(True, axis="y", alpha=0.2)

    ax.set_ticks(6)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    plt.tight_layout()

    if outdir is not None:
        outpath = Path(outdir / f"steady_state_bar.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


# ------------------------------------------------------------------
# Wrappers -- Key Timeseries
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

# --- Signalling ---
def plot_rho_balance_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="rho_balance",
        y_label="Δ = RhoA - RhoC",
        title="Rho balance over time",
        outdir=outdir,
        ax=ax,
    )

# --- Mechanics ---
def plot_squeeze_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="sf_squeeze",
        y_label=r"$T_{sf} \times \nu$",
        title="Stress fibre squeeze over time",
        outdir=outdir,
        ax=ax,
    )

def plot_tensions_balance_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="tension_balance",
        y_label=f"log_2(T_sf/T_cortex)",
        title="Cortex / Stress Fibre tension over time",
        outdir=outdir,
        ax=ax,
    )
