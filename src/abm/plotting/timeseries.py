# src/abm/plotting/timeseries.py
#
# Wrapper function for plotting ABM timeseries.
# Inlcudes wrapper for key metrics: AR, Rho balance, SF Squeeze, and cortex/sf tensions

from src.visualisation.line_plotter import plot_metric_timeseries, plot_timeseries_grid
from src.visualisation.style import PERTURBATION_COLOURS

# ------------------------------------------------------------------
# Core Metrics
# ------------------------------------------------------------------
def plot_ar_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="ar",
        y_label="AR",
        title="Aspect Ratio Over Time",
        outdir=outdir,
        ax=ax,
        order=None,
        colour_map=PERTURBATION_COLOURS,
    )

def plot_rho_balance_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="rho_balance",
        y_label="RhoA - RhoC",
        title="Rho Balance Over Time",
        outdir=outdir,
        ax=ax,
        order=None,
        colour_map=PERTURBATION_COLOURS,
    )

def plot_tensions_balance_timeseries(cell_df, outdir=None, ax=None):
    plot_metric_timeseries(
        cell_df,
        y_col="tension_balance",
        y_label=f"log_2(T_cortex/T_sf)",
        title="Cortex/SF Tension Ratio Over Time",
        outdir=outdir,
        ax=ax,
        order=None,
        colour_map=PERTURBATION_COLOURS,
    )

# ------------------------------------------------------------------
# Grid Wrappers for combined metrics
# ------------------------------------------------------------------
def plot_ar_and_rho_timeseries(cell_df, outdir=None):
    plot_timeseries_grid(
        df=cell_df, 
        metrics=[
            {"y_col": "ar", "y_label": "AR", "title": "Aspect Ratio"},
            {"y_col": "rho_balance", "y_label": "Δ = RhoA - RhoC", "title": "Rho Balance"}
        ],
        ncols=2, 
        title=None,#"AR and Rho Balance Over Time",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        outdir=outdir,
    )

def plot_rho_and_tension_timeseries(cell_df, outdir=None):
    plot_timeseries_grid(
        df=cell_df, 
        metrics=[
            {"y_col": "ar", "y_label": "AR", "title": "Aspect Ratio"},
            {"y_col": "rho_balance", "y_label": "Δ = RhoA - RhoC", "title": "Rho Balance"},
            {"y_col": "tension_balance", "y_label": "Cortex/SF Tension", "title": "Tension Balance"}
        ],
        ncols=2, 
        title="Rho and Tension Balance Over Time",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        outdir=outdir,
    )

def plot_key_metrics_timeseries(cell_df, outdir=None):
    plot_timeseries_grid(
        df=cell_df, 
        metrics=[
            {"y_col": "ar", "y_label": "AR", "title": "Aspect Ratio"},
            {"y_col": "rho_balance", "y_label": "Δ = RhoA - RhoC", "title": "Rho Balance"},
            {"y_col": "sf_squeeze", "y_label": "squeeze magnitude (T * v)", "title": "SF Squeeze Force"},
            {"y_col": "cortex_force_spread", "y_label": "cortex T polar/lateral", "title": "Cortex Tension Spread"}
        ],
        ncols=2, 
        title="Mechanics/Signalling During Elongation ",
        order=None,
        colour_map=PERTURBATION_COLOURS,
        outdir=outdir,
    )