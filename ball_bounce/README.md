# Sina "Bouncing Ball" Demo 

This is a simple demonstration workflow orchestrated by Maestro that uses Sina to collect data and produce visualizations of simulation inputs and outputs.

The simulation used here is `ball_bounce.py`, a (very) basic simulation of a "ball" (point) bouncing around in a 3D box. Maestro is used to generate sets of runs that share a(randomly chosen) gravity and starting position, but which differ by initial velocity. 

By default, each simulation runs for 20 seconds, or 400 ticks.

To emulate a "non-Sina-native" code, results are output as DSV and then ingested into Sina. Writing directly to Sina is possible (as well as faster and easier!) 

Visualizations are provided in the included Jupyter notebook.


## How to run

Run `setup.sh` to create a virtual environment with all necessary dependencies and install the jupyter kernel. 

Run `source ball_bounce_demo_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it)

Run `maestro run ball_bounce_suite.yaml --pgen pgen.py` to launch the studies. By default, this will run 20 simulations and ingest them all into the database. You can re-run the maestro command as many times as you like to continue adding runs.

Run `jupyter notebook` and open `visualization.ipynb` in the resulting browser window to access the visualizations.

Finally, run `teardown.sh` to delete the virtual environment, generated data, and the jupyter kernel. 


## Content overview

Starting files:
- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `ball_bounce_suite.yaml`: The maestro workflow description, containing all the information to run a set of ball bouncing simulations. Each set shares a starting position and gravity but differs on the initial velocities. 
- `pgen.py`: A custom parameter generator for Maestro, which will generate random starting conditions for each suite
- `visualization.ipynb`: A Jupyter notebook containing visualizations 
- `requirements.txt`: Requirements used to build the virtual environment

Files created by the demo:
- `output.sqlite`: A Sina datastore (here expressed as sqlite) containing all the results from all the suites run
- `output/`: The Maestro output location, containing all the files it generates 
- `ball_bounce_demo_venv/`: The virtual environment required to run the scripts, containing Sina, Maestro, and Jupyter
