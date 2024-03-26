"""Prepare a custom generator that will start a collection of balls in the same spot with differing velocities."""

import random
import uuid
import numpy as np
from trata import sampler
from maestrowf.datastructures.core import ParameterGenerator

seed = 1777
BOX_SIDE_LENGTH = 100
GRAVITY = .0981 # 100th of Earth's Gravity
coordinates = ["X", "Y", "Z"]
positions = ["{}_POS_INITIAL".format(coord) for coord in coordinates]
velocities = ["{}_VEL_INITIAL".format(coord) for coord in coordinates]
NUM_STUDIES = 1024

def get_custom_generator(env, **kwargs):

    p_gen = ParameterGenerator()
    LHCsampler = sampler.LatinHyperCubeSampler()

    # All balls in a single run share gravity, box side length, and group ID
    params = {"GRAVITY": {"values": [GRAVITY]*NUM_STUDIES,
                          "label": "GRAVITY.%%"},

              "BOX_SIDE_LENGTH": {"values": [BOX_SIDE_LENGTH]*NUM_STUDIES,
                                  "label": "BOX_SIDE_LENGTH.%%"},

              "GROUP_ID": {"values": ['47bcda']*NUM_STUDIES,
                           "label": "GROUP_ID.%%"},

              "RUN_ID": {"values": list(range(1, NUM_STUDIES+1)),
                         "label": "RUN_ID.%%"},

              "X_POS_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[47.5, 50.5]], seed=seed) for item in sublist],
                                "label": "X_POS_INITIAL.%%"},

              "Y_POS_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[48.5, 51.5]], seed=seed) for item in sublist],
                                "label": "Y_POS_INITIAL.%%"},

              "Z_POS_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[49.5, 52.5]], seed=seed) for item in sublist],
                                "label": "Z_POS_INITIAL.%%"},

              "X_VEL_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[5.10, 5.40]], seed=seed) for item in sublist],
                                "label": "X_VEL_INITIAL.%%"},

              "Y_VEL_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[4.75, 5.05]], seed=seed) for item in sublist],
                                "label": "Y_VEL_INITIAL.%%"},

              "Z_VEL_INITIAL": {"values":  [np.round(item, 4) for sublist in LHCsampler.sample_points(num_points=NUM_STUDIES, box=[[4.85, 5.15]], seed=seed) for item in sublist],
                                "label": "Z_VEL_INITIAL.%%"}
             }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    print("Preparing study set {} with gravity {}, starting position {}."
          .format(params["GROUP_ID"]["values"][0],
                  params["GRAVITY"]["values"][0],
                  tuple(params[x]["values"][0] for x in positions)))
    return p_gen
