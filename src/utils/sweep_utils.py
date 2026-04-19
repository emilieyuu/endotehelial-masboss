# src/utils/sweep_utils.py

import itertools

# Build parameter combinations from sweep config specs
def build_param_combinations(param_specs):
    """
    Build cartesian product of parameter combinations.

    param_specs: list of dicts with keys:
        - path: list[str]
        - values: list

    Returns: list[dict], where each dict maps tuple(path) -> value
    """
    paths = [tuple(spec["path"]) for spec in param_specs]
    values = [spec["values"] for spec in param_specs]

    combos = []
    for combo_vals in itertools.product(*values):
        combos.append(dict(zip(paths, combo_vals)))

    return combos

# Convert dict with tuple to dataframe row
def combo_to_row(combo):
    """
    Convert combo dict with tuple paths to flat dict for DataFrame rows.

    Example: {("mechanics", "k_base"): 2.0} becomes
        {"mechanics.k_base": 2.0}
    """
    return {".".join(path): value for path, value in combo.items()}

# Filename of specific sweep experiment for storing
def get_filename(model, target_sweeps, target_type):
    """
    Build output filename for selected sweep set.

    model: str – abm or bm
    target_sweeps: int – number of sweeps
    target_type: str – 1D or 2D
    """
    n_sweeps = len(target_sweeps) if target_sweeps else 0

    if target_type and not target_sweeps:
        filename = f"{model}_sweep_{target_type}"
    elif target_sweeps and n_sweeps == 1:
        filename = f"{model}_sweep_{target_sweeps[0]['name']}"
    elif target_sweeps and n_sweeps > 1:
        filename = f"{model}_sweep_{n_sweeps}_selected"
    else:
        filename = f"{model}_sweep_full"

    return filename
