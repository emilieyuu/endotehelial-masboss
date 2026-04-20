# boolean_models/experiments/lut_runner.py
#
# Thin wrapper for running the recruitment sweep used to build the ABM LUT.
# boolean_models/experiments/lut_runner.py

import pandas as pd

from src.boolean_model.runtime.model_loader import generate_ko_model
from src.boolean_model.experiments.parameter_sweep import build_ranges
from src.utils.file_utils import save_df_to_csv
from src.utils.sweep_utils import build_cartesian_product


def run_lut_sweep(base_model, sweep_cfg, sim_cfg, result_dir=None):
    """
    Run fixed 3D recruitment sweep for LUT generation.
    Always uses WT.
    """
    lut_cfg = sweep_cfg["lut"]

    # --- Build parameter ranges ---
    param_values = build_ranges(
        sweep_cfg,
        resolution=lut_cfg.get("resolution", "fine")
    )

    # Only keep LUT parameters
    param_values = {
        p: param_values[p]
        for p in lut_cfg["parameters"]
    }

    combos = build_cartesian_product(param_values)
    total = len(combos)
    print(f">>> INFO: Running LUT sweep ({len(combos)} combinations)")

    results = []

    for i, combo in enumerate(combos, start=1):
        # Progress tracker
        if i == 1 or i % 50 == 0 or i == total:
            print(f"    [{i}/{total}] running...")

        m_temp = base_model.copy()
        m_temp.update_parameters(**combo)

        res = m_temp.run()
        ss_df = res.get_last_nodes_probtraj()

        keys = list(combo.keys())
        vals = list(combo.values())

        ss_df["p1_name"], ss_df["p1_value"] = keys[0], vals[0]
        ss_df["p2_name"], ss_df["p2_value"] = keys[1], vals[1]
        ss_df["p3_name"], ss_df["p3_value"] = keys[2], vals[2]

        results.append(ss_df)

    lut_df = pd.concat(results, ignore_index=True)

    print(f">>> INFO: LUT sweep complete ({total} combinations)")

    # --- Save ---
    if result_dir is not None:
        save_df_to_csv(lut_df, result_dir, "rho_recruitment", ts=False)

    return lut_df