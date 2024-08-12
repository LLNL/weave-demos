import numpy as np
from trata.sampler import LatinHyperCubeSampler as LHS
from maestrowf.datastructures.core import ParameterGenerator

# Settings for the Latinhypercube sampler
Nruns = 15
test_box = [[0.3, 0.65], [0.85, 1.15]]
seed = 7
# Get space-filling samples for the multi-dimensional feature space
lhs_values = LHS.sample_points(box=test_box, num_points=Nruns, seed=seed)

# Separate and round the variables
atwood   = np.round(np.array(list(lhs_values[:, 0]), dtype=np.float),3)
velocity = np.round(np.array(list(lhs_values[:, 1]), dtype=np.float),3)
# Get a list of ints for the simulation random seed
simseed  = list(range(Nruns))


def get_custom_generator(env, **kwargs):

    p_gen = ParameterGenerator()

    params = {"ATWOOD": {"values": atwood,
                                "label": "ATWOOD.%%"},

              "VELOCITY_MAGNITUDE": {"values": velocity,
                                "label": "VEL.%%"},

              "SEED": {"values": simseed,
                                "label": "SEED.%%"},
             }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])


    return p_gen
