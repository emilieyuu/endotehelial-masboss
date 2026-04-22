## Run Order

The ABM depends on a precomputed lookup table generated from the Boolean model. Before running the ABM simulation for actual experiments, you must build that LUT once.

1. Generate the LUT:
python3 -m scripts.run_lut_sweep

This creates the recruitment lookup table in the LUT results directory and gives the ABM the Boolean-model outputs it needs at runtime.

2. Run the ABM simulation:
- single pertubation
python3 -m scripts.run_abm_sim single WT

- all perturbation: 
python3 -m scripts.run_abm_sim all


## Note
If you skip the LUT step, the ABM simulation will not have the required Boolean-model lookup data. You must run this first. 

CSV files of results for data analysis and steady-state cells figure is generated for full
six-perturbation runs only. The single cell run is used primarily for test purposes.
