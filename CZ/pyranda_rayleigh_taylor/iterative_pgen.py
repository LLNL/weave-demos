import numpy as np
from trata.sampler import LatinHyperCubeSampler as LHS
from trata.adaptive_sampler import ExpectedImprovementSampler as EIS
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.preprocessing import MinMaxScaler as MMS
from maestrowf.datastructures.core import ParameterGenerator


# Generate samples iteratively


def get_custom_generator(env, **kwargs):

    # This is the range of the 2 variables
    ls_test_box = [[0.3, 0.65], [0.85, 1.15]]

    Nruns = int(kwargs.get("NRUNS", env.find("NRUNS").value))
    Ncand = int(kwargs.get("NCAND", env.find("NCAND").value))
    curr_iter = int(kwargs.get("ITER", env.find("ITER").value))

    if curr_iter == 1:

        # Generating some initial inputs
        next_points = trata.sampler.LHS.sample_points(num_points=Nruns,
                                                         box=ls_test_box,
                                                         seed=20)
        # Separate and round the variables
        atwood   = np.round(np.array(list(lhs_values[:, 0]), dtype=np.float),3)
        velocity = np.round(np.array(list(lhs_values[:, 1]), dtype=np.float),3)

        first_iter = False  # somehow pass this back to maestro?

    else:

        # Something like this to get current points from the store
        # TODO args.name should be PREV_WORKSPACE from yaml
        current_inputs = next(store.find_ensembles(name=args.name))
        current_outputs = next(store.find_ensembles(name=args.name))

        # Scale inputs for the surrogate model
        scaler = MMS()
        scaled_inputs = scaler.fit_transform(current_inputs)

        # Training the default Gaussian process model from scikit learn
        surrogate_model = gpr().fit(scaled_inputs, current_outputs)

        # Generate some candidate points that adaptive sampling will choose from 
        candidate_points = trata.sampler.LHS.sample_points(num_points=Ncand,
                                                           box=ls_test_box,
                                                           seed=20)

        # Chooses samples from candidate points based on 2 criteria:
        # model error and output sensitivity
        next_points = EIS.sample_points(num_points=Nruns,
                                    cand_points=np_candidate_points,
                                    model=surrogate_model)

        # Separate and round the variables
        atwood   = np.round(np.array(list(next_points[:, 0]), dtype=np.float),3)
        velocity = np.round(np.array(list(next_points[:, 1]), dtype=np.float),3)

    params = {"ATWOOD": {"values": atwood,
                                "label": "ATWOOD.%%"},

              "VELOCITY_MAGNITUDE": {"values": velocity,
                                "label": "VEL.%%"},
             }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
