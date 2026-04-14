# abm/simulation.py
#
# Runtime orchestrator for one simulation.
#
# This class is responsible for executing a single simulation instance:
# cfg + lut + perturbation label -> time loop -> recorded outputs
#
# Responsibilities:
#   - Owns the runtime objects for one run (FlowField, CellAgent)
#   - Advances the model through discrete timesteps
#   - Records measurements into tabular outputs
#   - Optionally plots the final cell state

import time
import pandas as pd

from abm.flow_field import FlowField
from abm.cell_agent import CellAgent
from abm.analysis.cell_measurement import measure_cell, measure_springs, measure_nodes
from src.utils import require

class Simulation:
    """
    Execute one simulation run. 
    cfg: dict – fully resolved configuration
    lut: RhoLookupTable
    """

    def __init__(self, cfg, lut, perturbation="WT", plot=False):
        self.cfg = cfg
        self.lut = lut
        self.perturbation = perturbation
        self.plot = plot

        # --- Simulation controls ---
        sim_cfg = require(cfg, "simulation")
        self.dt = require(sim_cfg, "dt")
        self.n_steps = require(sim_cfg, "n_steps")
        self.detail_interval = require(sim_cfg, "detail_log_interval")

        # --- Runtime objects ---
        self.flow = FlowField(cfg)
        self.cell = CellAgent(
            cell_id=0,
            flow_axis=self.flow.direction,
            lut=lut,
            cfg=cfg,
        )

        # --- Output buffers ---
        self.cell_rows = []
        self.spring_rows = []
        self.node_rows = []

    # ------------------------------------------------------------------
    # Per-step Recording
    # ------------------------------------------------------------------
    def _record_step(self, step):
        """
        Build the three measurement records for one logged timestep.

        - cell-level logging: every step
        - spring/node logging: every detail_interval steps + final step 
        """
        t = round(step * self.dt, 2)
        exp_dict = {"step": step, "time": t, "perturbation": self.perturbation,}

        log_detail = (step % self.detail_interval == 0) or (step == self.n_steps - 1)

        self.cell_rows.append({**exp_dict, **measure_cell(self.cell)})

        if log_detail:
            self.spring_rows.extend([{**exp_dict, **r} for r in measure_springs(self.cell)])
            self.node_rows.extend([{**exp_dict, **r} for r in measure_nodes(self.cell)])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self):
        """
        Run the simulation for n_steps timesteps.

        Execution order:
          1. advance cell one step in the flow field
          2. record outputs for this timestep
          3. after the loop, compute final steady-state snapshots

        Returns: dict of: perturbation, cell/spring/node DataFrames, 
        steady-state summaries, final CellAgent object
        """
        print(f">>> INFO: Running perturbation: {self.perturbation} for {self.n_steps} steps.")
        t_start = time.perf_counter()

        for step in range(self.n_steps):
            self.cell.step(self.flow, dt=self.dt)
            self._record_step(step)

        # Steady-state snapshots.
        ss_exp_dict = {
            "step": self.n_steps - 1,
            "time": round((self.n_steps - 1) * self.dt, 2),
            "perturbation": self.perturbation,
        }

        cell_ss = {**ss_exp_dict, **measure_cell(self.cell)}
        spring_ss = [{**ss_exp_dict, **r} for r in measure_springs(self.cell)]
        node_ss = [{**ss_exp_dict, **r} for r in measure_nodes(self.cell)]

        runtime = time.perf_counter() - t_start
        print(
            f" {self.perturbation} ar={cell_ss['ar']:.3f} | "
            f"rho_balance={cell_ss['rho_balance']:.3f} | "
            f"rhoa={cell_ss['rhoa_mean']:.3f} | rhoc={cell_ss['rhoc_mean']:.3f} | "
            f"t={runtime:.1f}s"
        )

        if self.plot:
            self.plot_cell(title=f"{self.perturbation} (final)")

        return {
            "perturbation": self.perturbation,
            "cell_df": pd.DataFrame(self.cell_rows),
            "spring_df": pd.DataFrame(self.spring_rows),
            "node_df": pd.DataFrame(self.node_rows),
            "cell_ss": cell_ss,
            "spring_ss": spring_ss,
            "node_ss": node_ss,
            "cell": self.cell,
        }

    def plot_cell(self, title=""):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        polar_set = set(id(n) for n in self.cell.polar_nodes)

        for s in self.cell.springs:
            p1, p2 = s.node_1.pos, s.node_2.pos
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=1.2, alpha=0.6)

        for n in self.cell.nodes:
            colour = "red" if id(n) in polar_set else "steelblue"
            ax.scatter(*n.pos, c=colour, s=50, zorder=4)
            ax.text(n.pos[0] + 0.02, n.pos[1] + 0.02, str(n.id), fontsize=8)

        ax.scatter(*self.cell.centroid, marker="x", c="gray", s=60, zorder=5)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.set_title(title or "cell shape")
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.show()