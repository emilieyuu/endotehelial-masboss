# abm/plotting/cell_plotter.py
#
# Plotter for Cell objects.

import numpy as np
import matplotlib.pyplot as plt

from abm.analysis.cell_measurement import measure_cell

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
        transform=ax.transAxes, va="bottom", ha="right",fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.6),
    )

    # --- Axis formatting ---
    x, y = cell.positions[:, 0], cell.positions[:, 1]

    x_pad = 0.1 * max(x.max() - x.min(), 1.0)
    y_pad = 0.1 * max(y.max() - y.min(), 1.0)

    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")