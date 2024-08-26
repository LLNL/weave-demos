import os
import numpy as np
import kosh
import scipy.stats as sts
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from ibis import mcmc
import argparse
import matplotlib
matplotlib.use('Qt5Agg')

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--store", help="path to kosh/Sina store")
p.add_argument("--name", help="name for the ensemble of datasets to load", required=True)
p.add_argument("--nmodels", type=int, help="number of times to train models")
p.add_argument("--specroot", help="the specroot")

args = p.parse_args()

########################################################

#####   Reading in and transforming the data   ########

########################################################

# Connect to the Kosh store and read in datasets
store = kosh.connect(args.store)

# The experimental data was saved to the ensemble in our Kosh store
experiments_ensemble = next(store.find_ensembles(name="experiments"))
exp_uri = next(experiments_ensemble.find(mime_type="pandas/csv")).uri

# The simulation data was just generated with the workflow manager (Maestro or Merlin)
sim_ensemble = next(store.find_ensembles(name=args.name))

exp_uri = next(experiments_ensemble.find(mime_type="pandas/csv")).uri
exp_uri = os.path.join(args.specroot, exp_uri)
sim_uri = next(sim_ensemble.find(mime_type="pandas/csv")).uri

# Use the URI to read in datasets
rt_exp_data = np.genfromtxt(exp_uri, delimiter=',')
rt_sim_data = np.genfromtxt(sim_uri, delimiter=',')


# Separate inputs and outputs for experimental and simulation data
xexp = rt_exp_data[:, :2]
yexp = rt_exp_data[:, 2:]
xsim = rt_sim_data[:, :2]
ysim = rt_sim_data[:, 2:]

# The GP model and MCMC sampling perform better when inputs are scaled
# Using a min-max scaler from scikit-learn
scaler = MMS()
scaled_xsim = scaler.fit_transform(xsim)

# Also scale experimental data in the same way
scaled_xexp = scaler.transform(xexp)

########################################################

#####    Defining the MCMC sampling function   ########

########################################################


# Create the IBIS MCMC object
default_mcmc = mcmc.DefaultMCMC()

############# Input Section ####################

input_names = ['atwood_num', 'vel_mag']

# Calculate standard deviation for simulation input features
sim_std = np.std(scaled_xsim, axis=1)

ranges = [[.3, .8], [.7, 1.3]]

# Defining each input
# Scaled ranges are from 0.0 to 1.0 with
# We're using uninformative priors for both inputs
for i, name in enumerate(input_names):
    default_mcmc.add_input(name, 0.0, 1.0, sim_std[i], sts.uniform().pdf,
                           unscaled_low=ranges[i][0], unscaled_high=ranges[i][1], scaling='lin')

############## Output Section ##################

output_names = []
for i in range(nmodels):
    output_names.append('mix_width-%s' % i)

# Train the GP surrogates on simulation data
# At each time point specified by "nmodels"
surrogates = {}
for i in range(nmodels):
    ygp = ysim[:, i]
    name = output_names[i]
    surrogates[name] = GPR().fit(scaled_xsim, ygp)

# Calculate experimental mean and standard deviation
expMean = scaled_yexp.mean(axis=0)
expStd = scaled_yexp.std(axis=0)

for name, mean, std in zip(output_names, expMean, expStd):
    default_mcmc.add_output("RTexp", name, surrogates[name], mean, std, input_names)


# Run the MCMC chains to get samples approximating the posterior distribution
default_mcmc.run_chain(total=10000,         # The total number of samples in each chain
                       burn=500,            # The burn in period will be subtracted from total samples
                       every=2,             # Keep every # sample to reduce the correlation between samples
                       start={name: .5 for name in input_names},    # The starting point for sampling
                       prior_only=False,    # Whether to sample to estimate the priors or not
                       seed=15)             # Random seed for replication

chains = default_mcmc.get_chains(flattened=False, scaled=True)

# Plot the posterior distribution
outvar = list(default_mcmc.outputs.keys())[0]
post_pp = default_mcmc.posterior_predictive_plot(outvar, bins=10)

# Plot the MCMC sampling trace plots to check for good mixing
# Plot the informed distributions for each input
for input_n in input_names:
	plot = default_mcmc.trace_plot(input_name=input_n)
	hist_plot = default_mcmc.histogram_plot(input_name=input_n, bins=10, density=True, alpha=.5)
