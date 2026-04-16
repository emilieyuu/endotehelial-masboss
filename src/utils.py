# src/utils.py
#
from datetime import datetime
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
# Config Utils
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# File Access Utils
# ------------------------------------------------------------------
def save_df_to_csv(df, out_dir, base_name, ts=False):
    """
    Save DataFrame to CSV.

    df: pandas DataFrame.
    out_dir: Path, directory to save .csv file to.
    base_name: String, name for outgoing .csv file.
    timestamp: Boolean, whether or not to include timestamp in file name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ts:
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{base_name}_{time}.csv"
    else:
        file_name = f"{base_name}.csv"

    path = out_dir / file_name
    df.to_csv(path, index=False)
    print(f">>> INFO: Saved {path.name} to {out_dir}")
    return path

def safe_mean(lst):
    mean = float(np.mean(lst)) if lst else 0.0
    return round(mean, 3)