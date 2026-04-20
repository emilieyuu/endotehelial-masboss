# abm/scripts/run_abm_sim.py
#
# Thin entry script for the experiment.
#
# Example use:
#   python -m scripts.run_abm_sim single WT
#   python -m scripts.run_abm_sim all
#

import sys

from src.config_loader import load_abm_sim_cfg
from src.paths import LUT_DIR, ABM_SIM_RES_DIR
from src.abm.experiments.experiment_runner import ExperimentRunner

def main():
    sim_cfg = load_abm_sim_cfg()
    runner = ExperimentRunner(sim_cfg, LUT_DIR)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'single'

    if mode == 'single':
        perturbation = sys.argv[2] if len(sys.argv) > 2 else 'WT'
        result = runner.run_single(perturbation=perturbation)

        print(f">>> INFO: Finished single perturbation: {perturbation}")
        return result

    if mode == 'all':
        result = runner.run_all(result_dir=ABM_SIM_RES_DIR)

        print(">>> INFO: Finished all perturbations.")
        return result

    raise ValueError(
        f"Unknown mode '{mode}'. Use 'single' or 'all'."
    )


if __name__ == '__main__':
    main()