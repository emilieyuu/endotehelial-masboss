# abm/rho_lookup_table.py
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class RhoLookupTable: 
    def __init__(self, cfg, recruitment_dir):
        self.cfg = cfg

        # Load data
        path = recruitment_dir / cfg['files']['recruitment_csv']
        df = pd.read_csv(path).dropna(axis=1).round(3)
        print(f">>> DEBUG: Successfully loaded recruitment parameter sweep data.")
        
        # Interpolators
        self.rhoA_interp, self.rhoC_interp = self._build(df)

        # Rho at rest, used to update spring length
        self.rhoA_rest = float(self.rhoA_interp([0.0, 0.0, 0.0]))
        self.rhoC_rest = float(self.rhoC_interp([0.0, 0.0, 0.0]))
        print(f"LUT ready | rest: RhoA={self.rhoA_rest:.3f} RhoC={self.rhoC_rest:.3f}")

    def _build(self, df):
        # Extract protein name from parameter column names
        # e.g. "$DSP_recruitment" → "DSP"
        re_str  = r'\$?([A-Za-z0-9]+)_recruitment'
        param_map = {
            f'p{i}_value': df[f'p{i}_name'].str.extract(re_str)[0].iloc[0]
            for i in [1, 2, 3]
        }

        # Keep only input and output columns, rename inputs to protein names
        df_clean = (
            df[['p1_value', 'p2_value', 'p3_value', 'RhoA', 'RhoC']]
            .rename(columns=param_map)
        )
        print(f">>> DEBUG: Cleaned recruitment parameter sweep data.")


        # Sorted unique axis values for each protein
        axes = [np.sort(df_clean[p].unique()) for p in ['DSP', 'TJP1', 'JCAD']]

        # pivot_table maps each (DSP, TJP1, JCAD) combination to its correct
        rhoA_grid = (
            df_clean
            .pivot_table(index='DSP', columns=['TJP1', 'JCAD'], values='RhoA')
            .values
            .reshape(*map(len, axes))
        )
        rhoC_grid = (
            df_clean
            .pivot_table(index='DSP', columns=['TJP1', 'JCAD'], values='RhoC')
            .values
            .reshape(*map(len, axes))
        )

        if np.isnan(rhoA_grid).any() or np.isnan(rhoC_grid).any():
            raise ValueError("NaN in interpolation grid — missing combinations in sweep CSV. ")
 
        rhoA_interp = RegularGridInterpolator(
            axes, rhoA_grid,
            method='linear', bounds_error=False, fill_value=None
        )
        rhoC_interp = RegularGridInterpolator(
            axes, rhoC_grid,
            method='linear', bounds_error=False, fill_value=None
        )

        print(f">>> DEBUG: Successfully built interpolators")

        return rhoA_interp, rhoC_interp


    def query(self, p_dsp, p_tjp1, p_jcad):
        """
        Allows objects to query the Lookup Table.

        Query using junction protein activation probabilities.
        Returns activation/concentration of RhoA and RhoC. 
        """
        pt = [[
            float(np.clip(p_dsp,  0.0, 1.0)),
            float(np.clip(p_tjp1, 0.0, 1.0)),
            float(np.clip(p_jcad, 0.0, 1.0)),
        ]]

        return float(self.rhoA_interp(pt)), float(self.rhoC_interp(pt))