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

3. Run `maestro run ball_bounce_suite.yaml --pgen pgen.py` to launch the studies. By default, this will run 20 simulations and ingest them all into the database. You can re-run the maestro command as many times as you like to continue adding runs.

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

3. Follow the [WEAVE Workflow Toy Tutorial](https://lc.llnl.gov/weave/diagram.html). The numbered folders in the Content Overview section below correspond to the WEAVE Workflow Toy Tutorial steps.

4. Finally, run `teardown.sh` to delete the virtual environment, generated data, and the jupyter kernel. 

### Content overview

#### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `requirements.txt`: Requirements used to build the virtual environment
- `01_baseline_simulation/`
  - `baseline/`
    - `baseline.sh`: Runs four simulations to find the baseline simulation which will be used in the numerical resolution study and simulation ensemble.
    - `visualization_baseline.ipynb`: A Jupyter notebook to visualize the baseline simulation.
  - `num_res/`
    - `num_res.sh`: Runs three simulations at different time steps to see the effect on the baseline simulation resolution.
    - `visualization_num_res.ipynb`: A Jupyter notebook to visualize the baseline simulation numerical resolution.
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
  - `visualization_ensembles.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble data from the datastore.
  - `visualization_ensembles.py`: The Jupyter notebook converted to a Python script so the Maestro and Merlin workflow can process it.
- `06_surrogate_model/`
  - `ball_bounce_suite_maestro_surrogate_model.yaml`: The Maestro workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, post-processing the simulation ensemble, and creating the surrogate_model.
  - `ball_bounce_suite_merlin_surrogate_model.yaml`: The Merlin workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, post-processing the simulation ensemble, and creating the surrogate_model.
  - `visualization_surrogate_model.ipynb`: A Jupyter notebook to visualize the baseline simulation ensemble surrogate model data.
  - `visualization_surrogate_model.py`: The Jupyter notebook converted to a Python script so the Maestro and Merlin workflow can process it.

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