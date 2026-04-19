# src/utils/config_utils.py
#
# Reusable helpers for config manipulation and sweep setup.

import copy
import itertools


def copy_cfg(cfg):
    """Return a deep copy of config."""
    return copy.deepcopy(cfg)

def require(cfg, *keys):
    """
    Safely restrive a nested config value. 
    Raises KeyError with a clear path if any key is missing. 

    cfg: dictionary
    keys: separated sequence of dict keys
    """
    value = cfg

    # Traverse dictionary with keys
    for i, key in enumerate(keys):
        # Raise error if key is missing
        if not isinstance(value, dict) or key not in value: 
            path = '.'.join(keys[:i+1])
            raise KeyError(f"Missing required config key: {path}")
        # Index into dictionary
        value = value[key]
        
    return value

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
    cfg = copy_cfg(cfg_base)
    for path, value in combo.items():
        set_nested(cfg, list(path), value)
    return cfg
