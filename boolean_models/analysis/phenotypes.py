# boolean_models/analysis/phenotypes.py

import pandas as pd

def compute_delta(df, config):
    """ Compute delta: RhoC-RhoA balance. """
    rhoA = config['analysis']['nodes']['target_a']
    rhoC = config['analysis']['nodes']['target_b']
    #print(f"DEBUG: Computing delta = {rhoC} - {rhoA}")
    return df[rhoC] - df[rhoA]


def classify_phenotype(delta, config):
    """ Compute phenotype based on delta. """
    eps = config["analysis"]["eps"]
    labels = config["analysis"]["phenotypes"]

    if delta < -eps:
        return labels["failed"]
    elif delta > eps:
        return labels["hyper"]
    else:
        return labels["normal"]

