import numpy as np
from ibis import kosh_operators
import kosh
import h5py
import scipy.stats as sts
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor

input_names = ['atwood_num', 'vel_mag', 'time']
output_name = ['mix_width']
seed = 15
surrogate = GaussianProcessRegressor().fit(X, Y[:, i])
ranges = [[0, 1], [0, 1], [0, 1]]
flattened = False
scaled = True
fileName = "mytestfile.hdf5"
with h5py.File(fileName, "w") as f:
    f.create_dataset("inputs", data=X)
    f.create_dataset("outputs", data=Y)
    f.close()

store = kosh.connect("temp_testing.sql")
dataset = store.create("uq_data")
dataset.associate([fileName], 'hdf5')

result = kosh_operators.KoshMCMC(dataset["inputs"],
                                 method="default_mcmc",
                                 input_names=input_names,
                                 inputs_low=[0.0]*3,
                                 inputs_high=[1.0]*3,
                                 proposal_sigmas=[0.2]*3,
                                 prior=,
                                 outputs=dataset["outputs"],
                                 output_names=output_name,
                                 quantity='x',
                                 surrogate_model=surrogate,
                                 observed_values=[.5]*2,
                                 observed_std=[.1]*2,
                                 inputs=input_names,
                                 total_samples=10,
                                 burn=20,
                                 every=2,
                                 start={name: .5 for name in input_names},
                                 prior_only=True,
                                 seed=seed,
                                 flattened=False)[:]

chains = result.get_chains(flattened=flattened, scaled=scaled)
