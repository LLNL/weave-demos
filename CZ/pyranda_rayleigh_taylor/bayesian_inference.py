import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
#from ibis import kosh_operators
import kosh
import h5py
import scipy.stats as sts
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from ibis import mcmc
import argparse

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--store", help="path to kosh/Sina store")
p.add_argument("--name", help="name for the ensemble of datasets to load", required=True)
p.add_argument("--nmodels", type=int, help="number of times to train models")
p.add_argument("--specroot", help="the specroot")

args = p.parse_args()

# Define data for MCMC sampling
input_names = ['atwood_num', 'vel_mag']

output_names = []
for i in range(args.nmodels):
    output_names.append( 'mix_width-%s' % i)

seed = 15
ranges = [[.3, .8, .5], [.7, 1.3, .4] ] 
exp_dist = []
flattened = False
scaled = True

# Connect to the Kosh store and read in datasets
store = kosh.connect(args.store)

experiments_ensemble = next(store.find_ensembles(name="experiments"))
sim_ensemble = next(store.find_ensembles(name=args.name))
# dataset = store.open("uq_data")
exp_uri = next(experiments_ensemble.find(mime_type="pandas/csv")).uri
exp_uri = os.path.join(args.specroot, exp_uri)
sim_uri = next(sim_ensemble.find(mime_type="pandas/csv")).uri
rt_exp_data = np.genfromtxt(exp_uri, delimiter=',')
rt_sim_data = np.genfromtxt(sim_uri, delimiter=',')

# Separate inputs and outputs for experimental and simulation data
xexp = rt_exp_data[:,:2]
yexp = rt_exp_data[:,2:]
xsim = rt_sim_data[:,:2]
ysim = rt_sim_data[:,2:]

# Calculate experimental mean and standard deviation
expMean = yexp.mean(axis=0)
expStd  = yexp.std(axis=0)

# Build the GP surrogates for the number of models
surrogates = {}
for i in range(args.nmodels):
    ygp = ysim[:,i]
    name = output_names[i]
    surrogates[name] = GPR().fit(xsim, ygp)

# Create the IBIS MCMC object and define inputs and outputs
default_mcmc = mcmc.DefaultMCMC()
for name,rng in zip(input_names, ranges):
    default_mcmc.add_input(name, rng[0], rng[1], rng[2], sts.uniform().pdf)

for name,mean,std in zip(output_names, expMean, expStd):
    default_mcmc.add_output("RTexp", name, surrogates[name], mean , std, input_names )

# Run the MCMC chains to get samples approximating the posterior distribution
default_mcmc.run_chain(total=10000,
                       burn=500,
                       every=2,
                       start={name: .5 for name in input_names},
                       prior_only=False,
                       seed=seed)

chains = default_mcmc.get_chains(flattened=flattened, scaled=scaled)

# Plot the posterior distribution
outvar = list(default_mcmc.outputs.keys())[0]
# resid_plot = default_mcmc.residuals_plot(outvar, bins=10)
post_pp = default_mcmc.posterior_predictive_plot(outvar, bins=10)

# Plot the MCMC sampling trace plots to check for good mixing
# Plot the informed distributions for each input
for input_n in input_names:
	plot = default_mcmc.trace_plot(input_name=input_n)
	hist_plot = default_mcmc.histogram_plot(input_name=input_n, bins=10, density=True, alpha=.5)
