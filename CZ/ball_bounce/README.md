# Ball Bounce

This is a simple demonstration workflow orchestrated by Maestro that uses Sina to collect data and produce visualizations of simulation inputs and outputs.

The simulation used here is `ball_bounce.py`, a (very) basic simulation of a "ball" (point) bouncing around in a 3D box. Maestro is used to generate sets of runs that share a (randomly chosen) gravity and starting position, but which differ by initial velocity.

By default, each simulation runs for 20 seconds, or 400 ticks.

To emulate a "non-Sina-native" code, results are output as DSV and then ingested into Sina. Writing directly to Sina is possible (as well as faster and easier!)

Visualizations are provided in the included Jupyter notebook.

## How to run

1. Run `setup.sh` in the top directory to create a virtual environment with all necessary dependencies and install the jupyter kernel.

2. Run `source cz_tutorials_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it) and `cd` back into this directory.

3. Run `maestro run ball_bounce_suite.yaml --pgen pgen.py` to generate the studies, then y to launch. By default, this will run 10 simulations and ingest them all into the database. Once it completes, re-run the maestro command as many times as you like to continue adding runs. It should take around 2 minutes to finish each.

4. Run `jupyter notebook` and open `start_here.ipynb` and `visualization.ipynb` in the resulting browser window to access the visualizations.

## Content overview

### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `ball_bounce_suite.yaml`: The Maestro workflow description, containing all the information to run a set of ball bouncing simulations. Each set shares a starting position and gravity but differs on the initial velocities.
- `pgen.py`: A custom parameter generator for Maestro, which will generate random starting conditions for each suite
- `start_here.ipynb`: Simple introduction Jupyter notebook to start off with.
- `visualization.ipynb`: A Jupyter notebook containing visualizations

### Files created by the demo:

- `output.sqlite`: A Sina datastore (here expressed as sqlite) containing all the results from all the suites run
- `output/`: The Maestro output location, containing all the files it generates