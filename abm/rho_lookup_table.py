# abm/rho_lookup_table.py
#
#
#
import pandas as pd
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import LinearNDInterpolator


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
        self.rhoa_rest = float(self.rhoA_interp([0.0, 0.0, 0.0]))
        self.rhoc_rest = float(self.rhoC_interp([0.0, 0.0, 0.0]))
        print(f"LUT ready | rest: RhoA={self.rhoa_rest:.3f} RhoC={self.rhoc_rest:.3f}")

    def _build(self, df):
        df_filtered = df[['RhoA', 'RhoC',
                          'p1_name','p1_value', # DSP
                          'p2_name','p2_value', # TJP1
                          'p3_name','p3_value']] # JCAD
        
        long_df = pd.DataFrame({
            'RhoA': pd.concat([df_filtered['RhoA']]*3, ignore_index=True),
            'RhoC': pd.concat([df_filtered['RhoC']]*3, ignore_index=True),
            'name': pd.concat([df_filtered['p1_name'], df_filtered['p2_name'], df_filtered['p3_name']], ignore_index=True),
            'value': pd.concat([df_filtered['p1_value'], df_filtered['p2_value'], df_filtered['p3_value']], ignore_index=True),
        })

        recr_df = long_df.pivot_table(index=['RhoA','RhoC'],
                            columns='name',
                            values='value',
                            aggfunc='first').reset_index()

        points = recr_df[['$DSP_recruitment','$TJP1_recruitment','$JCAD_recruitment']].values

        rhoA_interp = LinearNDInterpolator(points, recr_df['RhoA'].values)
        rhoC_interp = LinearNDInterpolator(points, recr_df['RhoC'].values)

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
        rhoa = float(self.rhoA_interp(pt))
        rhoc = float(self.rhoC_interp(pt))
        
        # Fallback if outside convex hull
        if np.isnan(rhoa) or np.isnan(rhoc):
            rhoa = float(self.rhoA_nearest(pt))
            rhoc = float(self.rhoC_nearest(pt))
        
        return rhoa, rhoc