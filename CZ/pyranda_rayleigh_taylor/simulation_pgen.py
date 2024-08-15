import numpy as np
from trata.sampler import LatinHyperCubeSampler as LHS
from maestrowf.datastructures.core import ParameterGenerator


# Get space-filling samples for the multi-dimensional feature space
lhs_values = LHS.sample_points(box=test_box, num_points=Nruns, seed=lhs_seed)

# Separate and round the variables
atwood   = np.round(np.array(list(lhs_values[:, 0]), dtype=np.float),3)
velocity = np.round(np.array(list(lhs_values[:, 1]), dtype=np.float),3)

def get_custom_generator(env, **kwargs):

    p_gen = ParameterGenerator()

    params = {"ATWOOD": {"values": atwood,
                                "label": "ATWOOD.%%"},

              "VELOCITY_MAGNITUDE": {"values": velocity,
                                "label": "VEL.%%"},
             }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
