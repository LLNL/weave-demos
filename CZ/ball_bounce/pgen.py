"""Prepare a custom generator that will start a collection of balls in the same spot with differing velocities."""

import random
import uuid

from maestrowf.datastructures.core import ParameterGenerator

BOX_SIDE_LENGTH = 100
coordinates = ["X", "Y", "Z"]
positions = ["{}_POS_INITIAL".format(coord) for coord in coordinates]
velocities = ["{}_VEL_INITIAL".format(coord) for coord in coordinates]
NUM_STUDIES = 10


def get_custom_generator(env, **kwargs):
    p_gen = ParameterGenerator()
    # All balls in a single run share a gravity
    params = {"GRAVITY": {"values": [random.choice([0, 0.5, 2, 10])]*NUM_STUDIES,
                          "label": "GRAVITY.%%"},
              "BOX_SIDE_LENGTH": {"values": [BOX_SIDE_LENGTH]*NUM_STUDIES,
                                  "label": "BOX_SIDE_LENGTH.%%"},
              "GROUP_ID": {"values": [str(uuid.uuid4())[0:6]]*NUM_STUDIES,
                           "label": "GROUP_ID.%%"},
              "RUN_ID": {"values": list(range(1, NUM_STUDIES+1)),
                         "label": "RUN_ID.%%"}}
    # All balls in a single run start from the same location.
    for position in positions:
        params[position] = {"values": [random.randint(1, BOX_SIDE_LENGTH-1)]*NUM_STUDIES,
                            "label": "{}.%%".format(position)}
    # They differ by their initial velocities. 3 velocities apiece, 27 total runs.
    for velocity in velocities:
        params[velocity] = {"values": [random.randint(-10, 10) for _ in range(NUM_STUDIES)],
                            "label": "{}.%%".format(velocity)}

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    print("Preparing study set {} with gravity {}, starting position {}."
          .format(params["GROUP_ID"]["values"][0],
                  params["GRAVITY"]["values"][0],
                  tuple(params[x]["values"][0] for x in positions)))
    return p_gen
