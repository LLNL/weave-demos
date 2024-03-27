import random
import os
from maestrowf.datastructures.core import ParameterGenerator
from datetime import datetime
import pandas as pd
import yaml

TOTAL_SAMPLES = 2  # Number of samples for each iteration.


def get_samples(guess_file, output_path, next_guess_file=None):

    params = {}

    # pgen runs at each top level iteration _1,_2,_3, etc...
    if output_path[-2:] != '_1': # Not the first iteration

        last_iter_dir = int(output_path[-1])-1 # Grab from last iteration
        prev_output_path = output_path[:-1] +  str(last_iter_dir)
        guess_file = os.path.join(os.path.dirname(prev_output_path),'next_guess.csv')

        df_guess= pd.read_csv(guess_file)
        for col in df_guess.columns:
            params[col] = {}
            params[col]["values"] = [df_guess.iloc[0][col]] # Only has one row so will always be 0
            params[col]["label"] = f"{col}.%%"

        print(f'Output path: {output_path}')
        print(f'Last iteration: {last_iter_dir}')
        print(f'Previous iteration path: {prev_output_path}')
        print(f'Previous Iteration data:\n{df_guess}\n')

    elif next_guess_file is not None: # next_guess.csv file was passed in for continuation study

        df_next_guess= pd.read_csv(next_guess_file)
        for col in df_next_guess.columns:
            if col not in ['iteration', 'sim_end_res']:
                params[col] = {}
                params[col]["values"] = [df_next_guess.iloc[0][col]] # Only has one row so will always be 0
                params[col]["label"] = f"{col}.%%"

        print(f'Next Guess data:\n{df_next_guess}\n')

    else: # First iteration has guess and bounds

        with open(guess_file, "r") as stream:
            initial_guess = yaml.safe_load(stream)
            for key, val in initial_guess.items():
                params[key] = {}
                params[key]["values"] = [val['initial_guess']]
                params[key]["label"] = f"{key}.%%"

        print(f'Initial guess and bounds:\n{initial_guess}\n')

    return params


def get_custom_generator(env, **kwargs):

    initial_guess_and_bounds_file =  env.find("INITIAL_GUESS_AND_BOUNDS_PATH").value
    next_guess_file =  env.find("NEXT_GUESS_PATH").value if env.find("NEXT_GUESS_PATH") is not None else None
    output_path =  env.find("OUTPUT_PATH").value
    params = get_samples(initial_guess_and_bounds_file,output_path, next_guess_file)
    p_gen = ParameterGenerator()
    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
