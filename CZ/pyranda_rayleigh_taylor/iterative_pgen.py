import kosh
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
        next_points = LHS.sample_points(num_points=Nruns,
                                        box=ls_test_box,
                                        seed=20)
        # Separate and round the variables
        atwood   = np.round(np.array(list(next_points[:, 0]), dtype=float),3).tolist()
        velocity = np.round(np.array(list(next_points[:, 1]), dtype=float),3).tolist()

    else:

        # Connect to the Kosh store and read in simulation dataset
        store_name = kwargs.get("STORE", env.find("STORE").value)
        prev_workspace = kwargs.get("PREV_WORKSPACE", env.find("PREV_WORKSPACE").value)
        store = kosh.connect(store_name)
        sim_ensemble = next(store.find_ensembles(name=prev_workspace))
        sim_uri = next(sim_ensemble.find(mime_type="pandas/csv")).uri
        rt_sim_data = np.genfromtxt(sim_uri, delimiter=",")
        current_inputs = rt_sim_data[:,:2]
        current_outputs = rt_sim_data[:,2:]

        # Scale inputs for the surrogate model
        scaler = MMS()
        scaled_inputs = scaler.fit_transform(current_inputs)

        # Training the default Gaussian process model from scikit learn
        surrogate_model = gpr().fit(scaled_inputs, current_outputs)

        # Generate some candidate points that adaptive sampling will choose from 
        candidate_points = LHS.sample_points(num_points=Ncand,
                                             box=ls_test_box,
                                             seed=20)

        # Chooses samples from candidate points based on 2 criteria:
        # model error and output sensitivity
        next_points = EIS.sample_points(num_points=Nruns,
                                        cand_points=candidate_points,
                                        X=current_inputs,
                                        Y=current_outputs,
                                        model=surrogate_model)

        # Separate and round the variables
        atwood   = np.round(np.array(list(next_points[:, 0]), dtype=float),3).tolist()
        velocity = np.round(np.array(list(next_points[:, 1]), dtype=float),3).tolist()

    p_gen = ParameterGenerator()

    params = {"ATWOOD": {"values": atwood,
                                "label": "ATWOOD.%%"},

              "VELOCITY_MAGNITUDE": {"values": velocity,
                                "label": "VEL.%%"},
             }

    for key, value in params.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
