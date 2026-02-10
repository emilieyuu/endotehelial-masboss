# boolean_models/analysis/phenotypes.py

import pandas as pd

def compute_delta(prob_df):
    """ Store balance as difference between RhoC and RhoA"""
    return prob_df["RhoC"] - prob_df["RhoA"]

def classify_phenotype(prob_df, eps=0.25):
    """ Classify into discrete phenotype based on threshold. """
    delta = compute_delta(prob_df)

    return pd.DataFrame({
        "Failed": (delta < -eps).astype(float),
        "Hyper": (delta > eps).astype(float),
        "Normal": (abs(delta) <= eps).astype(float),
    }, index=prob_df.index)
