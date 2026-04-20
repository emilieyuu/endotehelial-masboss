# boolean_models/experiments/parameter_sweep.py

import numpy as np
import pandas as pd

from src.boolean_model.runtime.model_loader import generate_ko_model
from src.boolean_model.analysis.phenotypes import compute_delta, classify_phenotype
from src.utils.file_utils import save_df_to_csv
from src.utils.sweep_utils import (
    build_cartesian_product,
    get_selected_specs,
    get_filename,
)


# ------------------------------------------------------------------
# Range builder
# ------------------------------------------------------------------
def build_ranges(sweep_cfg, resolution="fine"):
    """
    Build parameter value ranges from Boolean sweep config.

    Returns dict: {param_name: np.ndarray}
    """
    ranges = sweep_cfg["ranges"]
    groups = sweep_cfg["groups"]
    rho_groups = groups["rhos"]
    junction_groups = groups["junctions"]

    param_dict = {}

    for p, info in ranges.items():
        step = info["step"]
        if resolution == "coarse":
            step *= 2

        values = np.arange(info["start"], info["stop"], step)

        if p == "recruitment":
            for group in junction_groups:
                param_dict[f"${group}_{p}"] = values
        else:
            for group in rho_groups:
                param_dict[f"${group}_{p}"] = values

    return param_dict


def build_param_values_for_spec(spec, sweep_cfg):
    """
    Resolve the actual parameter values used by one sweep spec.
    """
    all_values = build_ranges(sweep_cfg, resolution=spec.get("resolution", "fine"))
    return {p: all_values[p] for p in spec["parameters"]}


# ------------------------------------------------------------------
# Internal runners
# ------------------------------------------------------------------
def _run_single_combo(model, combo):
    """
    Run one Boolean model combo and return final steady-state probabilities.
    """
    m_temp = model.copy()
    m_temp.update_parameters(**combo)

    res = m_temp.run()
    return res.get_last_nodes_probtraj()


def run_1d_sweep_single(base_model, spec, perb_config, sweep_cfg):
    """
    Run one 1D Boolean sweep.
    """
    results = []
    param_values = build_param_values_for_spec(spec, sweep_cfg)
    combos = build_cartesian_product(param_values)

    for perb in spec["perturbations"]:
        print(f">>> INFO: Starting {spec['name']} for perturbation: {perb}")
        perb_model = generate_ko_model(base_model, perb_config[perb])

        for combo in combos:
            ss_df = _run_single_combo(perb_model, combo)

            p1 = list(combo.keys())[0]
            v1 = list(combo.values())[0]

            ss_df["p1_name"] = p1
            ss_df["p1_value"] = v1
            ss_df["p2_name"] = np.nan
            ss_df["p2_value"] = np.nan
            ss_df["perturbation"] = perb
            ss_df["exp_name"] = spec["name"]
            ss_df["type"] = spec["type"]

            results.append(ss_df)

    return pd.concat(results, ignore_index=True)


def run_2d_sweep_single(base_model, spec, perb_config, sweep_cfg):
    """
    Run one 2D Boolean sweep.
    """
    results = []
    param_values = build_param_values_for_spec(spec, sweep_cfg)
    combos = build_cartesian_product(param_values)

    for perb in spec["perturbations"]:
        print(f">>> INFO: Starting {spec['name']} for perturbation: {perb}")
        perb_model = generate_ko_model(base_model, perb_config[perb])

        for combo in combos:
            ss_df = _run_single_combo(perb_model, combo)

            keys = list(combo.keys())
            vals = list(combo.values())

            ss_df["p1_name"] = keys[0]
            ss_df["p1_value"] = vals[0]
            ss_df["p2_name"] = keys[1]
            ss_df["p2_value"] = vals[1]
            ss_df["perturbation"] = perb
            ss_df["exp_name"] = spec["name"]
            ss_df["type"] = spec["type"]

            results.append(ss_df)

    return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------
# Public runner
# ------------------------------------------------------------------
def run_sweeps(base_model, sweep_cfg, sim_cfg, target_sweeps=None, target_type=None, result_dir=None):
    """
    Run selected 1D / 2D Boolean sweeps.
    """
    perb_config = sim_cfg["perturbations"]
    all_specs = sweep_cfg["sweeps"]
    selected = get_selected_specs(all_specs, target_names=target_sweeps, target_type=target_type)

    if not selected:
        print(">>> ERROR: No matching sweep specifications found.")
        return pd.DataFrame()

    sweep_results = []

    for spec in selected:
        print(f"\n>>> INFO: Initialising {spec['name']} ({spec['type']})")

        if spec["perturbations"] == "all":
            spec = dict(spec)
            spec["perturbations"] = list(perb_config.keys())

        try:
            if spec["type"] == "1D":
                df = run_1d_sweep_single(base_model, spec, perb_config, sweep_cfg)
            elif spec["type"] == "2D":
                df = run_2d_sweep_single(base_model, spec, perb_config, sweep_cfg)
            else:
                print(f">>> ERROR: Unsupported sweep type '{spec['type']}.")
                continue

            df["delta"] = compute_delta(df, sim_cfg)
            df["phenotype"] = df["delta"].apply(lambda x: classify_phenotype(x, sim_cfg))
            sweep_results.append(df)

        except Exception as e:
            print(f">>> ERROR: Failed sweep {spec['name']}: {e}")

    if not sweep_results:
        print(">>> ERROR: No sweeps completed successfully.")
        return pd.DataFrame()

    full_df = pd.concat(sweep_results, ignore_index=True)

    if result_dir is not None:
        filename = get_filename(selected, target_type=target_type, prefix="param_sweep")
        save_df_to_csv(full_df, result_dir, filename, timestamp=False)

    return full_df