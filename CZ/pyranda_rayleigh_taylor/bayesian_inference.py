import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from ibis import kosh_operators
import kosh
import h5py
import scipy.stats as sts
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from ibis import mcmc
import argparse

input_names = ['atwood_num', 'vel_mag']

output_names = []
Nstations = 11
for i in range(Nstations):
    output_names.append( 'mix_width-%s' % i)

seed = 15
#surrogate = GaussianProcessRegressor().fit(X, Y[:, i])
ranges = [[.3, .8, .5], [.7, 1.3, .4] ] 
exp_dist = []
flattened = False
scaled = True

# store = kosh.connect("temp_testing.sql")c
# dataset = store.open("uq_data")
rt_exp_data = np.genfromtxt("rt_exp_data_atw.35_vel1.05.csv", delimiter=',')
rt_sim_data = np.genfromtxt("rt_sim_data.csv", delimiter=',')


# Process experimental data, get means/std at each time station
xexp = rt_exp_data[:,:2]
yexp = rt_exp_data[:,2:]
xsim = rt_sim_data[:,:2]
ysim = rt_sim_data[:,2:]

expMean = ysim.mean(axis=0)
expStd  = ysim.std(axis=0)

# Build the GP surrogates for the N stations
surrogates = {}
for station in range(Nstations):
    ygp = ysim[:,station]
    name = output_names[station]
    surrogates[name] = GPR().fit(xsim, ygp)


default_mcmc = mcmc.DefaultMCMC()
for name,rng in zip(input_names,ranges):
    default_mcmc.add_input(name, rng[0],rng[1], rng[2], sts.uniform().pdf)

for name,mean,std in zip(output_names,expMean,expStd):
    if "10" in name:
        default_mcmc.add_output("RTexp", name, surrogates[name], mean , std, input_names )


#result.add_output('mix_width', 'x', surrogate, .4 , .1, input_names)
default_mcmc.run_chain(total=10000,
                       burn=500,
                       every=2,
                       start={name: .5 for name in input_names},
                       prior_only=False,
                       seed=seed)



# result = kosh_operators.KoshMCMC(dataset["inputs"],
#                                  method="default_mcmc",
#                                  input_names=input_names,
#                                  inputs_low=[0.0]*3,
#                                  inputs_high=[1.0]*3,
#                                  proposal_sigmas=[0.2]*3,
#                                  prior=[sts.uniform.pdf]*Ndim,
#                                  outputs=dataset["outputs"],
#                                  output_names=output_name,
#                                  quantity='x',
#                                  surrogate_model=surrogate,
#                                  observed_values=dataset['exp_outputs'],
#                                  observed_std=[.1]*2,
#                                  inputs=input_names,
#                                  total_samples=10000,
#                                  burn=200,
#                                  every=5,
#                                  start={name: .5 for name in input_names},
#                                  prior_only=True,
#                                  seed=seed,
#                                  flattened=False)[:]

chains = default_mcmc.get_chains(flattened=flattened, scaled=scaled)
# diagnostics = default_mcmc.get_diagnostics(n_split=2, scaled=True)
# resid = default_mcmc.get_residuals()
# log_pp = default_mcmc.log_posterior_plot()

outvar = list(default_mcmc.outputs.keys())[0]
# resid_plot = default_mcmc.residuals_plot(outvar, bins=10)
post_pp = default_mcmc.posterior_predictive_plot(outvar, bins=10)

breakpoint()

for input_n in input_names:
	plot = default_mcmc.trace_plot(input_name=input_n)
	# corr_plot = default_mcmc.autocorr_plot(input_name=input_n, N=50, n_split=2)
	hist_plot = default_mcmc.histogram_plot(input_name=input_n, bins=10, density=True, alpha=.5)
