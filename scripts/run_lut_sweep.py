# scripts/run_lut_sweep.py
#
# Run the Boolean 3D recruitment sweep used to build the ABM LUT.

from src.config_loader import load_bm_sim_cfg, load_bm_sweep_cfg
from src.paths import LUT_DIR
from src.boolean_model.runtime.model_loader import load_base_model
from src.boolean_model.experiments.lut_runner import run_lut_sweep


def main():
    sim_cfg = load_bm_sim_cfg()
    sweep_cfg = load_bm_sweep_cfg()

    base_model = load_base_model(sim_cfg)

    run_lut_sweep(
        base_model=base_model,
        sweep_cfg=sweep_cfg,
        sim_cfg=sim_cfg,
        result_dir=LUT_DIR,
    )

    print(">>> INFO: LUT sweep complete")


if __name__ == "__main__":
    main()