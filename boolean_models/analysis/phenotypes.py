# boolean_models/analysis/phenotypes.py

import pandas as pd

def compute_delta(df, config):
    """ Compute delta: RhoC-RhoA balance. """
    rhoA = config['analysis']['nodes']['target_a']
    rhoC = config['analysis']['nodes']['target_b']
    print(f"DEBUG: Computing delta = {rhoC} - {rhoA}")
    return df[rhoC] - df[rhoA]

# Phenotype timeseries
# def classify_phenotype(prob_df, eps=0.25):
#     """ Classify into discrete phenotype based on threshold. """
#     #delta = compute_delta(prob_df)
#     delta = prob_df['delta']
#     pheno_df = pd.DataFrame({
#                     "Failed": (delta < -eps).astype(float),
#                     "Hyper": (delta > eps).astype(float),
#                     "Normal": (abs(delta) <= eps).astype(float),
#                 }, index=prob_df.index)

#     return pheno_df

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

