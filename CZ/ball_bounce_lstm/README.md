# Ball Bounce LSTM

This is a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) workflow that uses Merlin for workflow orchestration and Kosh for data management and post-processing.

The simulation used here is the same as the ball bounce demo but with added steps to train an LSTM model.

## How to run

1. Run `setup.sh` in the top directory to create a virtual environment with all necessary dependencies and install the jupyter kernel.

2. Run `source weave_demos_venv/bin/activate` to enter the virtual environment (you can `deactivate` when you've finished the demo to exit it) and `cd` back into this directory.

3. Follow the steps below.
   1. Run `merlin run ball_bounce_suite_merlin_lstm.yaml --pgen pgen_ensembles.py` and `merlin run-workers ball_bounce_suite_merlin_lstm.yaml`
      1. Change `NUM_STUDIES = 1024` to a smaller number depending on computer capability  (e.g. 64) in `pgen_ensembles.py`
   2. Run `visualization_lstm_kosh.ipynb`
      1. Update the lstm model and/or update `NUM_STUDIES` above to get more training samples

  * Note: If the notebook `visualization_lstm_kosh.ipynb` is to your liking, you can just run step 1 above and the updated notebook will automatically be exported as a script.

## Content overview

### Starting files:

- `ball_bounce.py`: The "simulation" script, containing all the logic for bouncing the ball
- `dsv_to_sina.py`: A bare-bones ingester that finds dsv files and inserts them into a Sina datastore
- `pgen_ensembles.py`: A custom parameter generator for Maestro and Merlin containing the uncertainty bounds of the parameters for the baseline simulation ensemble.
- `ball_bounce_suite_merlin_lstm.yaml`: The Merlin workflow description for running the baseline simulation ensemble, consolidating the data into a datastore, and training the LSTM RNN.
- `visualization_lstm_kosh.ipynb`: A Jupyter notebook to train the LSTM RNN.

### Files created by the demo:

- `ball-bounce-lstm`
  - `run-ball-bounce/`: Contains the baseline simulation ensemble.
  - `ingest-ball-bounce/`: Contains the baseline simulation ensemble datastore.
  - `lstm-ball-bounce/`: Contains the images created by the Jupyter notebook.