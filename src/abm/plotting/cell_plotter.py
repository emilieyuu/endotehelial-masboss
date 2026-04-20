# abm/plotting/cell_plotter.py
#
# Plotter for Cell objects.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.abm.analysis.cell_measurement import measure_cell

def _get_legend():
    return [
        Line2D([0], [0], color="tab:green", lw=1.8, label="Cortex"),
        Line2D([0], [0], color="tab:blue", lw=2.2, label="SF"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="lightpink", markeredgecolor="none",
               markersize=8, label="Lateral"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="deeppink", markeredgecolor="none",
               markersize=8, label="Polar"),
        Line2D([0], [0], marker="x", linestyle="None",
               color="tab:purple", markersize=8, markeredgewidth=1.6,
               label="Centroid"),
    ]

def plot_cells_grid(results, outdir=None):
    """
    Plot initial cell + all perturbation steady-state cells on a 2x4 grid.

    result: dict – result returned by ExperimentRunner.run_all().
    outdir : str/Path – optional directory to save figure.
    """
    results = results["results_by_perturbation"]

    # Cell to axis map
    grid_cells = [
        ("Initial", results["WT"]["cell_initial"], None),
        None,  # legend slot
        ("WT", results["WT"]["cell_final"], results["WT"]["cell_ss"]["time"]),
        ("JCAD_KO", results["JCAD_KO"]["cell_final"], results["JCAD_KO"]["cell_ss"]["time"]),
        ("DSP_KO", results["DSP_KO"]["cell_final"], results["DSP_KO"]["cell_ss"]["time"]),
        ("DSP_JCAD_DKO", results["DSP_JCAD_DKO"]["cell_final"], results["DSP_JCAD_DKO"]["cell_ss"]["time"]),
        ("TJP1_KO", results["TJP1_KO"]["cell_final"], results["TJP1_KO"]["cell_ss"]["time"]),
        ("TJP1_JCAD_DKO", results["TJP1_JCAD_DKO"]["cell_final"], results["TJP1_JCAD_DKO"]["cell_ss"]["time"]),
    ]

    # Shared axial limits
    all_cells = [c[1] for c in grid_cells if c is not None]

    x_all = np.concatenate([cell.positions[:, 0] for cell in all_cells])
    y_all = np.concatenate([cell.positions[:, 1] for cell in all_cells])

    x_pad = 0.05 * max(x_all.max() - x_all.min(), 1.0)
    y_pad = 0.05 * max(y_all.max() - y_all.min(), 1.0)

    x_lim = (x_all.min() - x_pad, x_all.max() + x_pad)
    y_lim = (y_all.min() - y_pad, y_all.max() + y_pad)

    # Fig layout
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    axes = axes.ravel()

    # Plot in specified panels
    for ax, item in zip(axes, grid_cells):
        # Legend
        if item is None: 
            ax.axis("off")
            ax.legend(handles=_get_legend(), loc="center", frameon=False, fontsize=10)
            continue

       # Plot cell
        label, cell, time = item
        plot_cell(ax, cell, perturbation=label, time=time, show_node_ids=False)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    
    # Shared title
    #fig.suptitle("Initial and Steady-States Phenotypes", fontsize=16, y=0.92)
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    if outdir is not None:
        outpath = Path(outdir / f"all_cells_ss.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')


    plt.show()
    plt.close



def plot_cell(ax, cell, perturbation=None, time=None, show_node_ids=True):
    """
    Plot a single Cell on the provided axis. 

    ax: matplotlib.axes.Axes – axis to draw on.
    time: float – optional timestep label.
    show_node_ids: bool – whether to label nodes
    """
    # --- Style ---
    clr_lateral = "lightpink"
    clr_polar = "deeppink"
    clr_sf = "tab:blue"
    clr_spring = "tab:green"
    clr_centroid = "tab:purple"

    # --- Springs ---
    for spring in cell.springs:
        p1, p2 = spring.node_1.pos, spring.node_2.pos
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=clr_spring, linewidth=1.5, alpha=0.8, zorder=1,
        )

    # --- Stress fibre --- 
    sf_up, sf_dn = cell.sf.node_up.pos, cell.sf.node_down.pos
    ax.plot(
        [sf_up[0], sf_dn[0]], [sf_up[1], sf_dn[1]],
        color=clr_sf, linewidth=2.0, alpha=0.9, zorder=2,
    )

    # --- Nodes ---
    polar_ids = {id(n) for n in cell.polar_nodes}

    for node in cell.nodes:
        is_polar = id(node) in polar_ids
        colour = clr_polar if is_polar else clr_lateral

        ax.scatter(
            node.pos[0], node.pos[1],
            s=45, color=colour, zorder=3,
        )

        if show_node_ids:
            ax.text(node.pos[0] + 0.05, node.pos[1] + 0.08, str(node.id), fontsize=7, zorder=4)

    # --- Centroid ---
    c = cell.centroid
    ax.scatter(
        c[0], c[1], marker="x", s=70, 
        color=clr_centroid, linewidths=1.6, zorder=4,
    )

    if show_node_ids:
        ax.text(c[0] - 1.2, c[1] + 0.25, "C", fontsize=8, weight="bold", zorder=5,)

    # --- Title --- 
    title = []
    if perturbation is not None: title.append(str(perturbation))
    if time is not None: title.append(f"t = {time:.1f}")

    if title:
        ax.set_title(" | ".join(title), fontsize=10)

    # --- Metrics ---
    metrics = measure_cell(cell)
    ax.text(
        0.97,0.03,
        f"AR = {metrics['ar']}\nΔ = {metrics['rho_balance']}",
        transform=ax.transAxes, va="bottom", ha="right",fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6),
    )

    # --- Axis formatting ---
    x, y = cell.positions[:, 0], cell.positions[:, 1]

    # x_pad = 0.1 * max(x.max() - x.min(), 1.0)
    # y_pad = 0.1 * max(y.max() - y.min(), 1.0)

    # ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    # ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")