# abm/experiment_runner.py
#
# High-level experiment manager.
#
# Responsibilities:
#   1. Own the base configuration for a family of runs
#   2. Build the Rho lookup table once per experiment session
#   3. Apply perturbation and user-specified config overrides
#   4. Run either a one or all perturbations
#   5. Aggregate and optionally save outputs

import copy
import pandas as pd

from abm.rho_lookup_table import RhoLookupTable
from abm.simulation import Simulation
from src.utils import require, save_df_to_csv


class ExperimentRunner:
    """
    Build configs and execute ABM experiments
    cfg: dict – base experiment configuration.
    lut_dir: pathlib.Path – directory with recruitment CSV for lookup table.
    """

    def __init__(self, cfg, lut_dir):
        # Experiment baseline.
        self.base_cfg = copy.deepcopy(cfg)

        # Build lookup table
        self.lut_dir = lut_dir
        self.lut = RhoLookupTable(self.base_cfg, lut_dir)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _copy_cfg(self):
        """Return fresh deep copy of base config."""
        return copy.deepcopy(self.base_cfg)

    def _apply_nested(self, cfg, updates):
        """Recursively apply nested config overrides."""
        for key, val in updates.items():
            if isinstance(val, dict):
                self._apply_nested(require(cfg, key), val)
            else:
                require(cfg, key)
                cfg[key] = val
        return cfg

    def _apply_perturbation(self, cfg, knockouts):
        """
        Apply a perturbation knockout overrides to config.

        knockouts: dict of {protein_name: {key: value}}, e.g.
            {'DSP': {'knocked_out': True}}
        """
        hill = require(cfg, "hill_params")
        for protein, override in knockouts.items():
            require(hill, protein).update(override)
        return cfg

    def _apply_user_overrides(self, cfg,
        cell_radius=None, n_nodes=None, cell_centroid=None,
        flow_direction=None, flow_magnitude=None,
        n_steps=None, dt=None, detail_log_interval=None,
    ):
        """Apply notebook/script-friendly overrides."""
        updates = {}

        # --- Cell overrides ---
        if cell_radius is not None:
            updates.setdefault("cell", {})["radius"] = cell_radius
        if n_nodes is not None:
            updates.setdefault("cell", {})["n_nodes"] = n_nodes
        if cell_centroid is not None:
            updates.setdefault("cell", {})["centroid"] = cell_centroid

        # --- Flow overrides ---
        if flow_direction is not None:
            updates.setdefault("flow", {})["direction"] = flow_direction
        if flow_magnitude is not None:
            updates.setdefault("flow", {})["magnitude"] = flow_magnitude

        # --- Simulation overrides ---
        if n_steps is not None:
            updates.setdefault("simulation", {})["n_steps"] = n_steps
        if dt is not None:
            updates.setdefault("simulation", {})["dt"] = dt
        if detail_log_interval is not None:
            updates.setdefault("simulation", {})["detail_log_interval"] = detail_log_interval

        return self._apply_nested(cfg, updates) if updates else cfg

    def build_cfg(self, perturbation=None, **user_kwargs):
        """Construct the fully resolved config for one run."""
        cfg = self._copy_cfg()

        if perturbation is not None:
            perturbations = require(cfg, "perturbations")
            knockouts = require(perturbations, perturbation)
            cfg = self._apply_perturbation(cfg, knockouts)

        cfg = self._apply_user_overrides(cfg, **user_kwargs)
        return cfg

    # ------------------------------------------------------------------
    # Single Perturbation Experiment Runner
    # ------------------------------------------------------------------
    def run_single(self, perturbation="WT", **user_kwargs):
        """
        Run one perturbation and return its result.

        Main entry point for notebook use.
        """
        cfg = self.build_cfg(perturbation=perturbation, **user_kwargs)

        sim = Simulation(cfg=cfg, lut=self.lut,perturbation=perturbation)

        return sim.run()

    # ------------------------------------------------------------------
    # Full Perturbation Experiment Runner
    # ------------------------------------------------------------------
    def run_all(self, result_dir=None, **user_kwargs):
        """
        Run all perturbations defined in the base config.

        Returns: Aggregated results containing per-perturbation ones and
            concatenated DataFrames across the full experiment set.
        """
        perturbations = require(self.base_cfg, "perturbations")

        cell_dfs, spring_dfs, node_dfs = [], [], []
        cell_ss_rows, spring_ss_rows, node_ss_rows = [], [], []
        results = {}

        for perturbation in perturbations:
            result = self.run_single(perturbation=perturbation, **user_kwargs)
            results[perturbation] = result

            cell_dfs.append(result["cell_df"])
            spring_dfs.append(result["spring_df"])
            node_dfs.append(result["node_df"])
            cell_ss_rows.append(result["cell_ss"])
            spring_ss_rows.extend(result["spring_ss"])
            node_ss_rows.extend(result["node_ss"])

        cell_ts_df = pd.concat(cell_dfs, ignore_index=True)
        spring_ts_df = pd.concat(spring_dfs, ignore_index=True)
        node_ts_df = pd.concat(node_dfs, ignore_index=True)
        cell_ss_df = pd.DataFrame(cell_ss_rows)
        spring_ss_df = pd.DataFrame(spring_ss_rows)
        node_ss_df = pd.DataFrame(node_ss_rows)

        if result_dir is not None:
            save_df_to_csv(cell_ts_df, result_dir, "abm_cell_timeseries", ts=False)
            save_df_to_csv(spring_ts_df, result_dir, "abm_spring_timeseries", ts=False)
            save_df_to_csv(node_ts_df, result_dir, "abm_node_timeseries", ts=False)
            save_df_to_csv(cell_ss_df, result_dir, "abm_cell_steady_state", ts=False)
            save_df_to_csv(spring_ss_df, result_dir, "abm_spring_steady_state", ts=False)
            save_df_to_csv(node_ss_df, result_dir, "abm_node_steady_state", ts=False)
            print(f">>> INFO: Results saved to {result_dir}")

        return {
            "results_by_perturbation": results,
            "cell_ss_df": cell_ss_df,
            "spring_ss_df": spring_ss_df,
            "node_ss_df": node_ss_df,
        }