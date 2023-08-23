# Sina "Bouncing Ball" Demo 

This is a simple demonstration workflow orchestrated by Maestro that uses Sina to collect data and produce visualizations of simulation inputs and outputs.

The simulation used here is `ball_bounce.py`, a (very) basic simulation of a "ball" (point) bouncing around in a 3D box. Maestro is used to generate sets of runs that share a (randomly chosen) gravity and starting position, but which differ by initial velocity.

By default, each simulation runs for 20 seconds, or 400 ticks.

To emulate a "non-Sina-native" code, results are output as DSV and then ingested into Sina. Writing directly to Sina is possible (as well as faster and easier!) 

Visualizations are provided in the included Jupyter notebook.

## Base Example

Base example on how to use some of the WEAVE Tools.

### How to run

1. Run `setup.sh` to create a virtual environment with all necessary dependencies and install the jupyter kernel. 

2. Run `source ball_bounce_demo_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it)

3. Run `maestro run ball_bounce_suite.yaml --pgen pgen.py` to generate the studies, then y to launch. By default, this will run 10 simulations and ingest them all into the database. Once it completes, re-run the maestro command as many times as you like to continue adding runs. It should take around 2 minutes to finish each.

4. Run `jupyter notebook` and open `visualization.ipynb` in the resulting browser window to access the visualizations.

5. Finally, run `teardown.sh` to delete the virtual environment, generated data, and the jupyter kernel. 

### Content overview

#### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `ball_bounce_suite.yaml`: The Maestro workflow description, containing all the information to run a set of ball bouncing simulations. Each set shares a starting position and gravity but differs on the initial velocities. 
- `pgen.py`: A custom parameter generator for Maestro, which will generate random starting conditions for each suite
- `visualization.ipynb`: A Jupyter notebook containing visualizations 
- `requirements.txt`: Requirements used to build the virtual environment

#### Files created by the demo:

- `output.sqlite`: A Sina datastore (here expressed as sqlite) containing all the results from all the suites run
- `output/`: The Maestro output location, containing all the files it generates 
- `ball_bounce_demo_venv/`: The virtual environment required to run the scripts, containing Sina, Maestro, and Jupyter

## WEAVE Workflow Example

WEAVE Workflow Example that uses all the WEAVE Tools.

### How to run

1. Run `setup.sh` to create a virtual environment with all necessary dependencies and install the jupyter kernel. 

2. Run `source ball_bounce_demo_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it)

3. Follow the [WEAVE Workflow Toy Tutorial](https://lc.llnl.gov/weave/diagram.html). The numbered folders in the Content Overview section below correspond to the WEAVE Workflow Toy Tutorial steps. If you don't have access to the tutorial, follow the steps below.
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

4. Finally, run `teardown.sh` to delete the virtual environment, generated data, and the jupyter kernel. 

### Content overview

#### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `requirements.txt`: Requirements used to build the virtual environment
- `01_baseline_simulation/`
  - `baseline/`
    - `baseline.sh`: Runs four simulations to find the baseline simulation which will be used in the numerical resolution study and simulation ensemble.
    - `visualization_baseline_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation using Sina.
    - `visualization_baseline_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation using Kosh.
  - `num_res/`
    - `num_res.sh`: Runs three simulations at different time steps to see the effect on the baseline simulation resolution.
    - `visualization_num_res_sina.ipynb`: A Jupyter notebook to visualize the baseline simulation numerical resolution using Sina.
    - `visualization_num_res_kosh.ipynb`: A Jupyter notebook to visualize the baseline simulation numerical resolution using Kosh.
    - `ball_bounce_15.py`: Sets the timestep to 15 ticks per second whereas baseline simulation is 20 ticks per second.
    - `ball_bounce_25.py`: Sets the timestep to 25 ticks per second whereas baseline simulation is 20 ticks per second.
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

#### Files created by the demo:

- `ball_bounce_demo_venv/`: The virtual environment required to run the scripts, containing Sina, Maestro, and Jupyter
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