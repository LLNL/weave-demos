"""Prepare a custom generator that will start a collection of balls in the same spot with differing velocities."""

import random
import uuid
import numpy as np

from maestrowf.datastructures.core import ParameterGenerator

BOX_SIDE_LENGTH = 100
GRAVITY = 9.81
coordinates = ["X", "Y", "Z"]
positions = ["{}_POS_INITIAL".format(coord) for coord in coordinates]
velocities = ["{}_VEL_INITIAL".format(coord) for coord in coordinates]
NUM_STUDIES = 1024

def get_custom_generator(env, **kwargs):
    p_gen = ParameterGenerator()
    # All balls in a single run share a gravity
    params = {"GRAVITY": {"values": [GRAVITY]*NUM_STUDIES,
                          "label": "GRAVITY.%%"},

              "BOX_SIDE_LENGTH": {"values": [BOX_SIDE_LENGTH]*NUM_STUDIES,
                                  "label": "BOX_SIDE_LENGTH.%%"},

              "GROUP_ID": {"values": [str(uuid.uuid4())[0:6]]*NUM_STUDIES,
                           "label": "GROUP_ID.%%"},

              "RUN_ID": {"values": list(range(1, NUM_STUDIES+1)),
                         "label": "RUN_ID.%%"},

              "X_POS_INITIAL": {"values":  np.round(np.random.normal(49.0, 0.5, NUM_STUDIES),4),
                                 "label": "X_POS_INITIAL.%%"},

              "Y_POS_INITIAL": {"values":  np.round(np.random.normal(50.0, 0.5, NUM_STUDIES),4),
                                 "label": "Y_POS_INITIAL.%%"},

              "Z_POS_INITIAL": {"values":  np.round(np.random.normal(51.0, 0.5, NUM_STUDIES),4),
                                 "label": "Z_POS_INITIAL.%%"},

              "X_VEL_INITIAL": {"values":  np.round(np.random.uniform(5.10, 5.40, NUM_STUDIES),4),
                                 "label": "X_VEL_INITIAL.%%"},

              "Y_VEL_INITIAL": {"values":  np.round(np.random.uniform(4.75, 5.05, NUM_STUDIES),4),
                                 "label": "Y_VEL_INITIAL.%%"},

              "Z_VEL_INITIAL": {"values":  np.round(np.random.uniform(4.85, 5.15, NUM_STUDIES),4),
                                 "label": "Z_VEL_INITIAL.%%"},                                        
            }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    print("Preparing study set {} with gravity {}, starting position {}."
          .format(params["GROUP_ID"]["values"][0],
                  params["GRAVITY"]["values"][0],
                  tuple(params[x]["values"][0] for x in positions)))
    return p_gen
