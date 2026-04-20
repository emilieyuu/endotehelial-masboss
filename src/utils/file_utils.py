# src/utils/file_utils.py

from datetime import datetime
from pathlib import Path
import pandas as pd

# DataFrame to csv
def save_df_to_csv(df, out_dir, base_name, suffix=None, ts=False):
    """
    Save DataFrame to CSV.

    df: pandas DataFrame.
    out_dir: Path, directory to save .csv file to.
    base_name: String, name for outgoing .csv file.
    suffix: optional string appended to filename.
    ts: Boolean, whether or not to include timestamp.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # clean suffix
    if suffix is not None:
        suffix = str(suffix).replace(" ", "_").replace("/", "_")

    if ts:
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parts = [base_name, suffix, time]
    else:
        parts = [base_name, suffix]

    # remove None parts
    parts = [p for p in parts if p is not None]
    file_name = "_".join(parts) + ".csv"

    path = out_dir / file_name
    df.to_csv(path, index=False)

    print(f">>> INFO: Saved {path.name} to {out_dir}")
    return path

# CSV Loader
def load_csv_to_df(in_dir, file_name, **read_csv_kwargs):
    """
    Load a CSV file into a pandas DataFrame.

    in_dir: Path or str – directory containing the file.
    file_name: str – name of the CSV file (with or without .csv extension).
    read_csv_kwargs: optional kwargs passed to pandas.read_csv().

    Returns: DataFrame
    """
    try:
        in_dir = Path(in_dir)

        # Ensure .csv extension
        if not file_name.endswith(".csv"):
            file_name = f"{file_name}.csv"

        path = in_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        df = pd.read_csv(path, **read_csv_kwargs)

        print(f">>> INFO: Loaded {path.name} from {in_dir}")
        return df

    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading {file_name} from {in_dir}: {e}")
    

# Save Figures
def save_figure(fig, outdir, title=None, filename=None):
    if outdir is None:
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if title is None:
            raise ValueError("Provide either title or filename to save figure.")
        filename = title.lower().replace(" ", "_").replace("/", "_") + ".png"

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")