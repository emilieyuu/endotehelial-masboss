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
        path = dir / self.cfg["files"]["recruitment_csv"]
        var_map = self.cfg["recruitment_var_maps"]

        df_raw = pd.read_csv(path)
        
        print(f">>> DEBUG: Successfully loaded recruitment parameter sweep data.")

        # Map MaBoSS variable names to respective probability names. 
        df_filtered = df_raw.dropna(axis=1).round(3)
        df_filtered["p1_name"] = df_filtered["p1_name"].map(var_map)

        # Create df indexed by protein activation probability. 
        df_index = df_filtered.set_index(["p1_name", "p1_value"])[["RhoA", "RhoC"]].reset_index()

        return df_index
    
    def _build_interpolators(self, df):
        interpolators = {}

        for p in df['p1_name'].unique():
            subset = df[df['p1_name'] == p].sort_values('p1_value')
            interpolators[p] = {"RhoA": interp1d(subset['p1_value'], subset['RhoA'], bounds_error=False, fill_value='extrapolate'),
                                "RhoC": interp1d(subset['p1_value'], subset['RhoC'], bounds_error=False, fill_value='extrapolate')}
            
        return interpolators

    def query(self, p_dsp, p_tjp1, p_jcad):
        vals = {
            "p_dsp": (self.interpolators["p_dsp"]["RhoA"](p_dsp), self.interpolators["p_dsp"]["RhoC"](p_dsp)), 
            "p_tjp1": (self.interpolators["p_tjp1"]["RhoA"](p_tjp1), self.interpolators["p_tjp1"]["RhoC"](p_tjp1)), 
            "p_jcad": (self.interpolators["p_jcad"]["RhoA"](p_jcad), self.interpolators["p_jcad"]["RhoC"](p_jcad))
        }

        p_rhoA = np.mean([v[0] for v in vals.values()])
        p_rhoC = np.mean([v[1] for v in vals.values()])

        return p_rhoA, p_rhoC