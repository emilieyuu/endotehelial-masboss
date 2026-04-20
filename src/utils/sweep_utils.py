# src/utils/sweep_utils.py

import itertools
import copy


def get_selected_specs(specs, target_names=None, target_type=None):
    """
    Filter sweep/spec list by optional names and/or type.
    """
    selected = specs

    if target_names:
        selected = [s for s in selected if s["name"] in target_names]
    if target_type:
        selected = [s for s in selected if s["type"] == target_type]

    return selected

def build_cartesian_product(param_values):
    """
    Build cartesian product from mapping of param -> values.

    Returns list[dict].
    """
    keys = list(param_values.keys())
    vals = list(param_values.values())
    
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*vals)
    ]

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

def get_filename(specs=None, target_type=None, prefix="param_sweep"):
    """
    Shared filename helper for saved sweep outputs.
    """
    n_specs = len(specs) if specs else 0

    if target_type and not specs:
        return f"{prefix}_{target_type}"
    if specs and n_specs == 1:
        return f"{prefix}_{specs[0]['name']}"
    if specs and n_specs > 1:
        return f"{prefix}_{n_specs}_selected"
    return f"{prefix}_full"

def set_nested(cfg, path, value):
    """
    Set nested config value in-place.

    path: list[str], e.g. ["mechanics", "k_base"]
    value: new value to replace current in cofig
    """
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return cfg


def apply_param_combo(cfg_base, combo):
    """
    Return deep-copied config with one parameter combination applied.

    combo: dict of {tuple(path): value}
    Example:
        {
            ("mechanics", "k_base"): 2.0,
            ("cortex", "a_drop"): 0.2,
        }
    """
    cfg = copy.deepcopy(cfg_base)
    for path, value in combo.items():
        set_nested(cfg, list(path), value)
    return cfg
