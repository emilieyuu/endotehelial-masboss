# abm/rho_lookup_table.py
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class RhoLookupTable: 
    def __init__(self, cfg, recruitment_dir):
        self.cfg = cfg
        self.index_table = self._build_indexed_table(recruitment_dir)
        self.interpolators = self._build_interpolators(self.index_table)


    def _build_indexed_table(self, dir):
        # Configure path and read csv in to dataframe 
        path = dir / self.cfg["files"]["recruitment_csv"]
        df_raw = pd.read_csv(path)
        print(f">>> DEBUG: Successfully loaded recruitment parameter sweep data.")

        # Extract protein name from paramater name.  
        df_filtered = df_raw.dropna(axis=1).round(3).copy()
        df_filtered["p1_name"] = df_filtered["p1_name"].str.extract(r'\$([A-Za-z0-9]+)_')

        # Create df indexed by protein activation probability. 
        df_index = df_filtered.set_index(["p1_name", "p1_value"])[["RhoA", "RhoC"]].reset_index()

        return df_index
    
    def _build_interpolators(self, df):
        # Initiate dict to hold interpolators
        interpolators = {rho: {} for rho in ["RhoA", "RhoC"]}

        # Build interpolator for RhoA and RhoC for each protein. 
        for p, subset in df.groupby('p1_name'):
            subset = subset.sort_values('p1_value')

            for rho in ["RhoA", "RhoC"]:
                interpolators[rho][p] = interp1d(
                    subset['p1_value'],
                    subset[rho],
                    bounds_error=False,
                    fill_value='extrapolate'
                )

        return interpolators

    def query(self, p_dsp, p_tjp1, p_jcad):
        """
        Allows objects to query the Lookup Table.

        Query using junction protein activation probabilities.
        Returns activation/concentration of RhoA and RhoC. 
        """
        probs = {"DSP": p_dsp, "TJP1": p_tjp1, "JCAD": p_jcad}

        vals = [
            (self.interpolators['RhoA'][k](v), self.interpolators['RhoC'][k](v))
            for k, v in probs.items()
        ]

        p_rhoA = np.mean([v[0] for v in vals])
        p_rhoC = np.mean([v[1] for v in vals])

        return p_rhoA, p_rhoC