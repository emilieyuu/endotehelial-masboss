import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PERTURBATION_ORDER = [
    "WT",
    "DSP_KO",
    "TJP1_KO",
    "JCAD_KO",
    "DSP_JCAD_DKO",
    "TJP1_JCAD_DKO",
]

PERTURBATION_TITLES = {
    "WT": "WT",
    "DSP_KO": "DSP KO",
    "TJP1_KO": "TJP1 KO",
    "JCAD_KO": "JCAD KO",
    "DSP_JCAD_DKO": "DSP-JCAD DKO",
    "TJP1_JCAD_DKO": "TJP1-JCAD DKO",
}


def plot_sweep_heatmaps(
    df, value_col, sweep_name="contractility_competition",
    x_col="stress_fibre.a_drop", y_col="cortex.a_drop",
    title=None, cmap="viridis", center=None, cbar_label=None, outdir=None,
):
    """
    Plot 2D sweep heatmaps for one metric, one panel per perturbation.
    """
    df = df.copy()
    df = df[df["sweep_name"] == sweep_name]

    if df.empty:
        raise ValueError(f"No rows found for sweep_name='{sweep_name}'")

    # Shared colour scale
    vals = df[value_col].dropna().values
    vmin = np.min(vals)
    vmax = np.max(vals)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes = axes.ravel()

    mappable = None

    for ax, perturbation in zip(axes, PERTURBATION_ORDER):
        sub = df[df["perturbation"] == perturbation]

        if sub.empty:
            ax.axis("off")
            continue

        pivot = sub.pivot(index=y_col, columns=x_col, values=value_col)
        pivot = pivot.sort_index().sort_index(axis=1)

        x_vals = pivot.columns.values
        y_vals = pivot.index.values
        z = pivot.values

        if center is None:
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            lim = max(abs(vmin - center), abs(vmax - center))
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=center - lim,
                vmax=center + lim,
            )

        mappable = im

        ax.set_title(PERTURBATION_TITLES[perturbation], fontsize=10)

        ax.set_xticks(np.arange(len(x_vals)))
        ax.set_xticklabels([f"{x:.1f}" for x in x_vals], rotation=45, ha="right", fontsize=8)

        ax.set_yticks(np.arange(len(y_vals)))
        ax.set_yticklabels([f"{y:.1f}" for y in y_vals], fontsize=8)

        ax.set_xlabel("SF contractility (a_drop)", fontsize=9)
        ax.set_ylabel("Cortex contractility (a_drop)", fontsize=9)

    fig.subplots_adjust(right=0.88, wspace=0.25, hspace=0.35)

    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    if cbar_label is not None:
        cbar.set_label(cbar_label)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.98)

    if outdir is not None:
        outpath = Path(outdir / f"{value_col}_contractility_sweep.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()

# ------------------------------------------------------------------
# Wrappers
# ------------------------------------------------------------------
def plot_ar_heatmaps(df, outdir=None):
    plot_sweep_heatmaps(
        df,
        value_col="ar",
        title="Aspect ratio across contractility sweep",
        cmap="viridis",
        cbar_label="Aspect ratio",
        outdir=outdir,
    )


def plot_rho_balance_heatmaps(df, outdir=None):
    plot_sweep_heatmaps(
        df,
        value_col="rho_balance",
        title="Rho balance across contractility sweep",
        cmap="RdBu_r",
        center=0.0,
        cbar_label="Δ = RhoC - RhoA",
        outdir=outdir,
    )


def plot_sf_squeeze_heatmaps(df, outdir=None):
    plot_sweep_heatmaps(
        df,
        value_col="sf_squeeze",
        title="Stress fibre squeeze across contractility sweep",
        cmap="magma",
        cbar_label="SF squeeze",
        outdir=outdir,
    )


def plot_cortex_balance_heatmaps(df, outdir=None):
    df['cortex_T_spread'] = np.log2(df["cortex_T_ratio"] + 1e-10)
    plot_sweep_heatmaps(
        df,
        value_col="cortex_T_spread",
        title="Cortex tension balance across contractility sweep",
        cmap="RdBu_r",
        center=0.0,
        cbar_label=r"log$_2$(polar / lateral cortex T)",
        outdir=outdir,
    )