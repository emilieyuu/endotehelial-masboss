# abm/experiments/parameter_sweep.py
#
# Run ABM parameter sweeps: 1D or 2D.
# Sweep experiment configured in config/abm_sweep.yaml
#
# Output format:
#   one row per perturbation per parameter combination

import pandas as pd

from src.utils.file_utils import save_df_to_csv
from src.utils.sweep_utils import (
    apply_param_combo,
    build_param_combinations,
    combo_to_row,
    get_filename,
    get_selected_specs,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _run_combo(runner, cfg_base, sweep, combo):
    """
    Run all perturbations for one parameter combination.
    Returns long-format DataFrame: one row per perturbation for this combination.
    """
    cfg = apply_param_combo(cfg_base, combo)

    # Rebuild runner for this combo
    temp_runner = runner.__class__(cfg, runner.lut_dir)

    result = temp_runner.run_all()
    df = result["cell_ss_df"].copy()

    # Attach metadata
    df["sweep_name"] = sweep["name"]
    df["type"] = sweep["type"]

    for k, v in combo_to_row(combo).items():
        df[k] = v

    return df

# ------------------------------------------------------------------
# Generic sweep runner (1D / 2D)
# ------------------------------------------------------------------
def run_sweep_single(runner, cfg_base, sweep):
    """
    Run one sweep specification (1D or 2D).
    Runs sweep and return results. Does not save to csv.
    """
    combos = build_param_combinations(sweep["parameters"])
    results = []

    total = len(combos)

    print(f">>> INFO: Starting {sweep['type']} sweep: {sweep['name']}")
    print(f">>> INFO: {total} parameter combinations")

    for i, combo in enumerate(combos, 1):
        if i == 1 or i % 10 == 0 or i == total:
            print(f"    [{i}/{total}] {combo_to_row(combo)}")

        try:
            df = _run_combo(runner, cfg_base, sweep, combo)
            results.append(df)
        except Exception as e:
            print(f">>> ERROR: Sweep combo failed: {combo_to_row(combo)} | {e}")

    if not results:
        print(f">>> ERROR: No successful runs for sweep {sweep['name']}")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------
# Full combined sweep runner
# ------------------------------------------------------------------
def run_sweeps(runner, sweep_cfg, result_dir=None, target_sweeps=None, target_type=None):
    """
    Run selected ABM sweeps.

    runner : ExperimentRunner – Base experiment runner.
    sweep_cfg : dict – abm_sweep.yaml.
    target_sweeps : list[str] – optional list of sweep experiment to run.
    target_type : str – optional filter by sweep type ("1D", "2D").
    result_dir : Path – optional directory to save output CSV.

    Returns: DataFrame – long-format combined sweep result.
    """
    """
    Run selected ABM sweeps.
    """
    all_sweeps = sweep_cfg["sweeps"]

    # Shared filtering logic
    selected = get_selected_specs(
        all_sweeps,
        target_names=target_sweeps,
        target_type=target_type,
    )

    if not selected:
        print(">>> ERROR: No matching sweep specifications found.")
        return pd.DataFrame()

    sweep_results = []

    for sweep in selected:
        print(f"\n>>> INFO: Initialising sweep: {sweep['name']} ({sweep['type']})")

        try:
            df = run_sweep_single(runner, runner.base_cfg, sweep)

            if not df.empty:
                sweep_results.append(df)

        except Exception as e:
            print(f">>> ERROR: Failed sweep {sweep['name']}: {e}")

    if not sweep_results:
        print(">>> ERROR: No sweeps completed successfully.")
        return pd.DataFrame()

    full_df = pd.concat(sweep_results, ignore_index=True)

    if result_dir is not None:
        filename = get_filename(selected, target_type, prefix="abm_sweep")
        save_df_to_csv(full_df, result_dir, filename, ts=False)
        print(f">>> INFO: Sweep results saved to {result_dir}")

    return full_df