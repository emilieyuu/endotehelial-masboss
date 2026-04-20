# boolean_model/runtime/model_loader.py

import maboss
from src.paths import MABOSS_DIR


def load_base_model(sim_cfg):
    """
    Load configured MaBoSS model and apply runtime parameters.
    """
    model_cfg = sim_cfg["model"]

    bnd_path = MABOSS_DIR / model_cfg["bnd"]
    cfg_path = MABOSS_DIR / model_cfg["cfg"]

    model = maboss.load(str(bnd_path), str(cfg_path))
    model.param["max_time"] = sim_cfg["simulation"]["max_time"]
    model.param["sample_count"] = sim_cfg["simulation"]["sample_count"]

    return model


def generate_ko_model(base_model, mutations):
    """
    Return copy of base model with KO mutations applied.
    """
    model = base_model.copy()

    for node, state in mutations.items():
        model.mutate(node, state)

    return model