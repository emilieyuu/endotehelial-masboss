# abm/experiments/parameter_sweep.py
#
# Run ABM parameter sweeps: 1D or 2D.
# Sweep experiment configured in config/abm_sweep.yaml
#
# Output format:
#   one row per perturbation per parameter combination

import pandas as pd

from src.utils.config_utils import apply_param_combo
from src.utils.file_utils import save_df_to_csv
from src.utils.sweep_utils import build_param_combinations, combo_to_row, get_filename

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _run_combo(runner, cfg_base, sweep, combo):
    """
    Run all perturbations for one parameter combination.

    Returns long-format DataFrame: one row per perturbation for this combination.
    """
    cfg = apply_param_combo(cfg_base, combo)

    # Reuse runner logic, but with overridden config for this combo
    # Create a temporary runner so lookup table reflects this config if needed.
    temp_runner = runner.__class__(cfg, runner.lut_dir)

    result = temp_runner.run_all()
    cell_ss_df = result["cell_ss_df"].copy()

    # Attach sweep metadata + parameter values
    cell_ss_df["sweep_name"] = sweep["name"]
    cell_ss_df["type"] = sweep["type"]

    combo_row = combo_to_row(combo)
    for key, value in combo_row.items():
        cell_ss_df[key] = value

    return cell_ss_df


# ------------------------------------------------------------------
# Sweep runners
# ------------------------------------------------------------------
def run_1d_sweep_single(runner, cfg_base, sweep):
    """
    Run one 1D sweep specification.

    Returns one long-format DataFrame with one row per perturbation
    per parameter value.
    """
    combos = build_param_combinations(sweep["parameters"])
    results = []

    print(f">>> INFO: Starting 1D sweep: {sweep['name']}")
    print(f">>> INFO: {len(combos)} parameter combinations")

    for i, combo in enumerate(combos, 1):
        combo_str = combo_to_row(combo)
        print(f"    [{i}/{len(combos)}] {combo_str}")

        try:
            combo_df = _run_combo(runner, cfg_base, sweep, combo)
            results.append(combo_df)
        except Exception as e:
            print(f">>> ERROR: Sweep combo failed: {combo_str} | {e}")

    if not results:
        print(f">>> ERROR: No successful runs for sweep {sweep['name']}")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def run_2d_sweep_single(runner, cfg_base, sweep):
    """
    Run one 2D sweep specification.

    Returns one long-format DataFrame with one row per perturbation
    per parameter combination.
    """
    combos = build_param_combinations(sweep["parameters"])
    results = []

    print(f">>> INFO: Starting 2D sweep: {sweep['name']}")
    print(f">>> INFO: {len(combos)} parameter combinations")

    for i, combo in enumerate(combos, 1):
        combo_str = combo_to_row(combo)
        print(f"    [{i}/{len(combos)}] {combo_str}")

        try:
            combo_df = _run_combo(runner, cfg_base, sweep, combo)
            results.append(combo_df)
        except Exception as e:
            print(f">>> ERROR: Sweep combo failed: {combo_str} | {e}")

    if not results:
        print(f">>> ERROR: No successful runs for sweep {sweep['name']}")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ------------------------------------------------------------------
# Full combined sweep runner
# ------------------------------------------------------------------
def run_sweeps(runner, sweep_cfg, target_sweeps=None, target_type=None, result_dir=None):
    """
    Run selected ABM sweeps.

    runner : ExperimentRunner – Base experiment runner.
    sweep_cfg : dict – abm_sweep.yaml.
    target_sweeps : list[str] – optional list of sweep experiment to run.
    target_type : str – optional filter by sweep type ("1D", "2D").
    result_dir : Path – optional directory to save output CSV.

    Returns: DataFrame – long-format combined sweep result.
    """
    all_sweeps = sweep_cfg["sweeps"]

    # Filter by name
    if target_sweeps:
        all_sweeps = [s for s in all_sweeps if s["name"] in target_sweeps]

    # Filter by type
    if target_type:
        all_sweeps = [s for s in all_sweeps if s["type"] == target_type]

    if not all_sweeps:
        print(">>> ERROR: No matching sweep specifications found.")
        return pd.DataFrame()

    sweep_results = []

    for sweep in all_sweeps:
        print(f"\n>>> INFO: Initialising sweep: {sweep['name']} ({sweep['type']})")

        try:
            if sweep["type"] == "1D":
                df = run_1d_sweep_single(runner, runner.base_cfg, sweep)
            elif sweep["type"] == "2D":
                df = run_2d_sweep_single(runner, runner.base_cfg, sweep)
            else:
                print(f">>> ERROR: Unknown sweep type '{sweep['type']}' for {sweep['name']}")
                continue

            if not df.empty:
                sweep_results.append(df)

        except Exception as e:
            print(f">>> ERROR: Failed sweep {sweep['name']}: {e}")

    if not sweep_results:
        print(">>> ERROR: No sweeps completed successfully.")
        return pd.DataFrame()

    full_sweep_df = pd.concat(sweep_results, ignore_index=True)

    if result_dir is not None:
        filename = get_filename("abm", all_sweeps, target_type)
        save_df_to_csv(full_sweep_df, result_dir, filename, ts=False)
        print(f">>> INFO: Sweep results saved to {result_dir}")

    return full_sweep_df