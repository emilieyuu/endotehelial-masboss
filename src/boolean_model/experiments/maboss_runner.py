# src/boolean_model/experiments/maboss_simulation.py
import pandas as pd

from src.boolean_model.analysis.phenotypes import compute_delta, classify_phenotype
from src.boolean_model.runtime.model_loader import generate_ko_model
from src.utils.file_utils import save_df_to_csv


def run_maboss_sim(base_model, cfg, result_dir=None):
    """
    Run MaBoSS simulations for all configured perturbations.

    param base_model: MaBoSS model to copy and mutate per perturbation.
    param cfg: dict — simulation config containing perturbation and analysis settings.
    param result_dir: Path — optional output directory for CSV exports.

    return full_perb_df: DataFrame — Full timeseries for each perturbation.
    return ss_df: DataFrame — Final steady-state probabilities only.
    """
    perbs_dict = cfg["perturbations"]

    perbs = []

    for name, mutation in perbs_dict.items():
        print(f">>> INFO: Running perturbation: {name}")

        m = generate_ko_model(base_model, mutation)

        res = m.run()
        prob_df = res.get_nodes_probtraj().rename_axis("t").reset_index()
        prob_df["perturbation"] = name

        balance_df = prob_df.copy()
        balance_df["delta"] = compute_delta(balance_df, cfg)

        phenotype_df = balance_df.copy()
        phenotype_df["phenotype"] = balance_df["delta"].apply(
            lambda x: classify_phenotype(x, cfg)
        )

        perbs.append(phenotype_df)

    print(">>> INFO: All MaBoSS perturbation simulations completed successfully")

    full_perb_df = pd.concat(perbs, ignore_index=True)

    ss_mask = full_perb_df.groupby("perturbation")["t"].idxmax()
    ss_df = full_perb_df.loc[ss_mask].reset_index(drop=True)

    if result_dir is not None:
        save_df_to_csv(full_perb_df, result_dir, "bm_sim_timeseries")
        save_df_to_csv(ss_df, result_dir, "bm_sim_steady_state")

    return full_perb_df, ss_df
