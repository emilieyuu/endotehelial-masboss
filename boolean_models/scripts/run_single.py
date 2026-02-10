
import maboss
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# --------------------------------------------------
# Paths (always relative to THIS FILE, not cwd)
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
MODELS_DIR = PROJECT_ROOT / "boolean_models" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "boolean_models"


# --------------------------------------------------
# Load model
# --------------------------------------------------
model_file = MODELS_DIR / "rho.bnd"
cfg_file   = MODELS_DIR / "rho_base.cfg"

res_name = "run_single"
scenario_result_dir = RESULTS_DIR / "rho_test" / res_name

def main():
    print(PROJECT_ROOT)
    
    model = maboss.load(str(model_file), str(cfg_file))
    shutil.copy(
        cfg_file,
        scenario_result_dir / "used_parameters.cfg"
    )
    print("Model and base configuration loaded")


    if scenario_result_dir.exists():
        shutil.rmtree(scenario_result_dir)

    scenario_result_dir.mkdir(parents=True, exist_ok=True)
    print("Created directory to store results")

    # --------------------------------------------------
    # Run simulation
    # --------------------------------------------------
    res = model.run()

    # --------------------------------------------------
    # Node probability results
    # --------------------------------------------------
    prob_df = res.get_nodes_probtraj()
    prob_df["delta"] = prob_df["RhoC"] - prob_df["RhoA"]

    nodes = prob_df.rename_axis("t").reset_index()
    nodes.to_csv(
        scenario_result_dir / f"{res_name}_nodes_probtraj.csv",
        index=False
    )

    print(f"Stored node prob traj in {scenario_result_dir}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    prob_df["RhoA"].plot(ax=ax, label="RhoA")
    prob_df["RhoC"].plot(ax=ax, label="RhoC")
    prob_df["delta"].plot(ax=ax, label="delta")

    ax.set_title("RhoA / RhoC activation probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        scenario_result_dir / f"{res_name}_rho_balance.png",
        dpi=300
    )

    plt.close(fig)
    print("Stored balance graph")


if __name__ == "__main__":
    main()

