# Ball Bounce VVUQ

This is a Verification, Validation, and Uncertainty & Quantification workflow that uses all of WEAVE's tools in tandem. The user can use Maestro or Merlin for workflow orchestration and Sina or Kosh for data management and post-processing.

The simulation used here is the same as the ball bounce demo but with added steps to perform VVUQ.

## How to run

1. Run `setup.sh` in the top directory to create a virtual environment with all necessary dependencies and install the jupyter kernel.

2. Run `source weave_demos_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it) and `cd` back into this directory.

3. Follow the [Ball Bounce VVUQ](https://lc.llnl.gov/weave/tutorials/CZ/bouncing_ball_vvuq/1_baseline_simulation.html) tutorial. The numbered folders in the Content Overview section below correspond to the WEAVE Workflow Toy Tutorial steps. If you don't have access to the tutorial, follow the steps below.
   1. `sh 01_baseline_simulation/baseline/baseline.sh`
   2. Run `01_baseline_simulation/baseline/visualization_baseline_sina.ipynb`
   3. `sh 01_baseline_simulation/num_res/num_res.sh`
   4. Run `01_baseline_simulation/num_res/visualization_num_res_sina.ipynb`
   5. `maestro run 04_manage_data/ball_bounce_suite_maestro_data_management.yaml --pgen 02_uncertainty_bounds/pgen_ensembles.py`
      1. Change `NUM_STUDIES = 1024` to a smaller number depending on computer capability  (e.g. 64) in `02_uncertainty_bounds/pgen_ensembles.py`
   6. Run `05_post-process_data/visualization_ensembles_sina.ipynb`
      1. Change `convergence = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]` to `convergence = [1, 2, 4, 8, 16, 32, 64]` to match step above
   7. Run `06_surrogate_model/visualization_surrogate_model.ipynb`

  * Note: If the notebooks for `05_post-process_data` and `06_surrogate_model` are to your liking, you can just run `maestro run 06_surrogate_model/ball_bounce_suite_maestro_surrogate_model.yaml --pgen 02_uncertainty_bounds/pgen_ensembles.py` for step 5 above and everything will run without having to run the `05_post-process_data` and `06_surrogate_model` notebooks individually. **Be sure to export these updated notebooks as Python scripts since that is what Maestro/Merlin are looking for.**

## Content overview

### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `01_baseline_simulation/`
  - `baseline/`
    - `baseline.sh`: Runs four simulations to find the baseline simulation which will be used in the numerical resolution study and simulation ensemble.
    - `ball_bounce_experiment.py`: The "experiment" script, containing all the logic for bouncing the ball
    - `visualization_baseline_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation using Sina.
    - `visualization_baseline_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation using Kosh.
  - `num_res/`
    - `num_res.sh`: Runs three simulations at different time steps to see the effect on the baseline simulation resolution.
    - `visualization_num_res_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation numerical resolution using Sina.
    - `visualization_num_res_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation numerical resolution using Kosh.
    - `ball_bounce_15.py`: Sets the time step to 15 ticks per second whereas baseline simulation is 20 ticks per second.
    - `ball_bounce_25.py`: Sets the time step to 25 ticks per second whereas baseline simulation is 20 ticks per second.
- `02_uncertainty_bounds/`
  - `pgen_ensembles.py`: A custom parameter generator for Maestro and Merlin containing the uncertainty bounds of the parameters for the baseline simulation ensemble.
- `03_simulation_ensembles/`
  - `ball_bounce_suite_maestro_simulation_ensembles.yaml`: The Maestro workflow description for running the baseline simulation ensemble.
  - `ball_bounce_suite_merlin_simulation_ensembles.yaml`: The Merlin workflow description for running the baseline simulation ensemble.
- `04_manage_data/`
  - `ball_bounce_suite_maestro_data_management.yaml`: The Maestro workflow description for running the baseline simulation ensemble and consolidating the data into a datastore.
  - `ball_bounce_suite_merlin_data_management.yaml`: The Merlin workflow description for running the baseline simulation ensemble and consolidating the data into a datastore.
- `05_post-process_data/`
  - `ball_bounce_suite_maestro_post-process_data.yaml`: The Maestro workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, and post-processing the simulation ensemble.
  - `ball_bounce_suite_merlin_post-process_data.yaml`: The Merlin workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, and post-processing the simulation ensemble.
  - `visualization_ensembles_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble data from the datastore using Sina.
  - `visualization_ensembles_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble data from the datastore using Kosh.
  - `visualization_ensembles_sina.py`: The Jupyter notebook converted to a Python script so the Maestro and Merlin workflow can process it using Sina.
- `06_surrogate_model/`
  - `ball_bounce_suite_maestro_surrogate_model.yaml`: The Maestro workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, post-processing the simulation ensemble, and creating the surrogate_model.
  - `ball_bounce_suite_merlin_surrogate_model.yaml`: The Merlin workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, post-processing the simulation ensemble, and creating the surrogate_model.
  - `visualization_surrogate_model_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble surrogate model data using Sina.
  - `visualization_surrogate_model_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble surrogate model data using Kosh.
  - `visualization_surrogate_model_sina.py`: The Jupyter notebook converted to a Python script so the Maestro and Merlin workflow can process it using Sina.

### Files created by the demo:

- `01_baseline_simulation/`
  - `baseline/`
    - `data/`: Contains the dsv files for the individual baseline simulations and the sqlite datastore which combines them all.
    - `images/`: Contains the images created by the Jupyter notebook.
  - `num_res/`
    - `data/`: Contains the dsv files for the individual baseline numerical resolution simulations and the sqlite datastore which combines them all.
    - `images/`: Contains the images created by the Jupyter notebook.
- `03_simulation_ensembles/`
  - `data/`: Contains the baseline simulation ensemble.
- `04_manage_data/`
  - `data/`: Contains the baseline simulation ensemble datastore.
- `05_post-process_data/`
  - `images/`: Contains the images created by the Jupyter notebook.
- `06_surrogate_model/`
  - `images/`: Contains the images created by the Jupyter notebook.