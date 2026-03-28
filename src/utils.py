# src/utils.py
from datetime import datetime
from pathlib import Path


def save_df_to_csv(df, out_dir, base_name, timestamp=False):
    """
    Save DataFrame to CSV.

    df: pandas DataFrame.
    out_dir: Path, directory to save .csv file to.
    base_name: String, name for outgoing .csv file.
    timestamp: Boolean, whether or not to include timestamp in file name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if timestamp:
        ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{base_name}_{ts}.csv"
    else:
        file_name = f"{base_name}.csv"

    path = out_dir / file_name
    df.to_csv(path, index=False)
    print(f">>> INFO: Saved {path.name} to {out_dir}")
    return path