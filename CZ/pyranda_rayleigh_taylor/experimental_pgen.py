import numpy as np
from maestrowf.datastructures.core import ParameterGenerator

# Generate sample points for the parameters from normal distributions
# for our fake experimental data
Nexperiments = 8

# Atwood distribution samples
atwood = np.round(0.48 + 0.08 * np.random.standard_normal(Nexperiments), 3)

# Velocity magnitude samples
velocity = np.round(0.97 + 0.1 * np.random.standard_normal(Nexperiments), 3)

# Get a list of ints for the simulation random seed
simseed  = list(range(Nexperiments))


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