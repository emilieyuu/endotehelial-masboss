# src/abm/plotting/sweeps.py
#
# Wrapper function for plotting sweep results.


from src.visualisation.sweep_plotter import plot_sweep_lines
from src.visualisation.style import PERTURBATION_COLOURS

# ------------------------------------------------------------------
# 1D Sweep Wrappers
# ------------------------------------------------------------------
def plot_ar_k_sweep(df, outdir=None):
    plot_sweep_lines(
        df,
        x_col="mechanics.k_base",
        y_col="ar",
        group_col="perturbation",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        title=None,
        x_label="base stiffness (k_base)",
        y_label="AR",
        outdir=outdir,
    )

def plot_rho_k_sweep(df, outdir=None):
    plot_sweep_lines(
        df,
        x_col="mechanics.k_base",
        y_col="rho_balance",
        group_col="perturbation",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        title="Rho Balance Across Stiffness Sweep",
        x_label="base stiffness (k_base)",
        y_label="RhoA-RhoC",
        outdir=outdir,
    )


def plot_sf_squeeze_k_sweep(df, outdir=None):
    plot_sweep_lines(
        df,
        x_col="mechanics.k_base",
        y_col="sf_squeeze",
        group_col="perturbation",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        title="Stress Fibre Squeeze Across Stiffness Sweep",
        x_label="base stiffness (k_base)",
        y_label="SF squeeze magnitude",
        outdir=outdir,
    )

def plot_cortex_spread_k_sweep(df, outdir=None):
    plot_sweep_lines(
        df,
        x_col="mechanics.k_base",
        y_col="cortex_force_spread",
        group_col="perturbation",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        title="Cortex Tension Spread Across Stiffness Sweep",
        x_label="base stiffness (k_base)",
        y_label=r"log$_2$(polar / lateral T)",
        outdir=outdir,
    )