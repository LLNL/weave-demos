#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

import sina.datastores.sql as sina_sql
import sina.utils
from sina.datastore import create_datastore
from sina.visualization import Visualizer

from trata import sampler as sm, composite_samples as cs, adaptive_sampler as ad
from ibis import mcmc as mc, plots
import os
import sys

if sys.argv[1] == '-f':  # Running as notebook
    spec_root = ''
    get_ipython().run_line_magic('matplotlib', 'notebook')
else:
    spec_root = sys.argv[1]  # Running as script
  


# Uncertainty Estimation via MCMC
# ===============================
# In our bouncing ball example, we want to estimate the distribution of the final x position of the ball given a starting position and velocity. 
# 
# An MCMC algorithm allows to simulate a probability distribution by constructing a Markov chain with the desired distribution as its stationary distribution. The MCMC algorithm iteratively updates the Markov chain based on the transition probability from one state to another state. Eventually, the chain attains the state of equilibrium when the joint probability distribution for the current state approaches the stationary distribution. The parameters that lead to the stationary distribution are considered as the model parameters learnt for the particular training image. 
# 
# Each state of a Markov chain is obtained by sampling a probability distribution. Among various sampling techniques, Metropolis algorithm and Gibbs sampler are two most well-known ones. See the appendix at the bottom for details. 
# 

# ## Sampling for Uncertainty Quantification
# 
# The Gaussian process model requires specification of two kinds of inputs:
# 
# 1. $x$ = $(x_1,x_2,...,x_p)$ denotes inputs that are under the control of (or are observable by) the experimenter in both the field experiments and the simulator runs. In our bouncing ball example, there are $p=3$ inputs of this type:
#     - $x_1=R=1$, the radius of the ball
#     - $x_2=D=1$, the density of the ball
#     - $x_3=C=0.1$, the coefficient of drag
#     
# <br>
# 
# 2. $\theta$ = $(\theta_1,\theta_2,...,\theta_q)$ denotes inputs to the simulator that are needed to estimate using the experimental data. These $\theta$ could correspond to real physical quantities or could be parameters of the simulator code. In our bouncing ball example, there are $q=2$ inputs of this type: 
#     - $\theta_1=P=[0,100]$, the initial position of ball
#     - $\theta_2=V=[-10,10]$, the initial velocity of the ball
# 
# 
# $\theta$ parameters of the bouncing ball model need to be appropriately sampled in order to build an accurate surrogate that honors the ground truth.
# 
# ### Load Data
# To get started we need to load our simulation data by creating a Sina DataStore.
# 

# In[ ]:


# Ensembles Initialization
database = os.path.join(spec_root, '../04_manage_data/data/ensembles_output.sqlite')
target_type = "csv_rec"
datastore = create_datastore(database)
recs = datastore.records
vis = Visualizer(datastore)


# Baseline Initialization
database_baseline = os.path.join(spec_root, '../01_baseline_simulation/baseline/data/baseline_output.sqlite')
datastore_baseline = create_datastore(database_baseline)
recs_baseline = datastore_baseline.records
vis_baseline = Visualizer(datastore_baseline)

print("Sina is ready!")


# In[ ]:


base_groups = set(x["group_id"]["value"] for x in recs_baseline.get_data(["group_id"]).values())

print("So far we've run {} experiment group(s) each with 4 studies in our baseline simualtion.".format(len(base_groups)))
print("We queried our database and found the following groups: {}".format(base_groups))


# We want to check the range of our target variable, the `x_pos_final`, across our data groups.

# In[ ]:


_= vis_baseline.create_histogram("x_pos_final", interactive=True)


# In[ ]:


import statistics as sts

base_groups = set(x["group_id"]["value"] for x in recs_baseline.get_data(["group_id"]).values())


# storing the simulation data locally to calculate descriptive statistics
x_pos_final_test, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel = [], [], [], [], [], [], []

# Printing parameter table in Markdown
print('| rec.id | x_pos_final | x_pos_initial | y_pos_initial | z_pos_initial | x_vel_initial | y_vel_initial | z_vel_initial |')
print('| --- | --- | --- | --- | --- | --- | --- | --- |')
# accessing and sampling the data for our surrogate model
for group in base_groups:
    id_pool = list(recs_baseline.find_with_data(group_id=group))
    for _, rec_id in enumerate(id_pool):
        rec = recs_baseline.get(rec_id)
        data = rec.data_values
        x_pos_final_test.append(data["x_pos_final"])
        x_pos.append(data["x_pos_initial"])
        y_pos.append(data["y_pos_initial"])
        z_pos.append(data["z_pos_initial"])
        x_vel.append(data["x_vel_initial"])
        y_vel.append(data["y_vel_initial"])
        z_vel.append(data["z_vel_initial"])
        
        print('|', rec_id,
              '|', data["x_pos_final"],
              '|', data["x_pos_initial"],
              '|', data['y_pos_initial'],
              '|', data['z_pos_initial'],
              '|', data['x_vel_initial'],
              '|', data['y_vel_initial'],
              '|', data['z_vel_initial'],
              '|'
             )

# calculating sigma for our output
xf_std = sts.stdev(x_pos_final_test)

# calculating sigmas for each input
sig_x = sts.stdev(x_pos)
sig_y = sts.stdev(y_pos)
sig_z = sts.stdev(z_pos)
sig_vx = sts.stdev(x_vel)
sig_vy = sts.stdev(y_vel)
sig_vz = sts.stdev(z_vel)


# ### Markov chain Monte Carlo
# Now that we've generated data, we need to load the inputs and output into our Markov chain Monte Carlo. 

# In[ ]:


from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd

groups = set(x["group_id"]["value"] for x in recs.get_data(["group_id"]).values())

# instantiating the MCMC object
vanilla_exp = mc.DefaultMCMC()

# accessing and sampling the data for our surrogate model
# we need to allocate training data from our previously run simulations to train a machine learning surrogate model
# we will be training our model on x_pos_final from our simulation data as well as position and velocity samples

x_pos_final_train, x_pos_train, y_pos_train, z_pos_train, x_vel_train, y_vel_train, z_vel_train = [], [], [], [], [], [], []

for group in groups:
    id_pool = list(recs.find_with_data(group_id=group))
    for _, rec_id in enumerate(id_pool):
        rec = recs.get(rec_id)
        data = rec.data_values
        
        # our feature training data
        x_pos_train.append(data["x_pos_initial"])
        y_pos_train.append(data["y_pos_initial"])
        z_pos_train.append(data["z_pos_initial"])
        x_vel_train.append(data["x_vel_initial"])
        y_vel_train.append(data["y_vel_initial"])
        z_vel_train.append(data["z_vel_initial"])

        # our target training data
        x_pos_final_train.append(data["x_pos_final"])


# our sampled feature training data
features_train = pd.DataFrame([x_pos_train, y_pos_train, z_pos_train, x_vel_train, y_vel_train, z_vel_train])
features_train = features_train.transpose()
print(features_train)

x_pos_final_train = pd.Series(x_pos_final_train)
print(x_pos_final_train)

# we have opted to use a Guassian Process Regressor as our surrogate model for this problem
surrogate_model = GaussianProcessRegressor()

# fitting the surrogate model using our parameter samples and the x position final simulated for those parameters
surrogate_model.fit(features_train, x_pos_final_train)        

# adding the target testing data and our fitted model as output to the MCMC for each observed x_pos_final
for _ in x_pos_final_test:
    vanilla_exp.add_output('output', 'x_pos_final', surrogate_model, _, xf_std, 
            ['x_pos_initial', 'y_pos_initial','z_pos_initial','x_vel_initial','y_vel_initial','z_vel_initial'])
    
_= vis.create_histogram("x_pos_final", interactive=True)


# Adding Inputs to our MCMC

# In[ ]:


# adding the input data to our MCMC
vanilla_exp.add_input('x_pos_initial', min(x_pos), max(x_pos), sig_x)
vanilla_exp.add_input('y_pos_initial', min(y_pos), max(y_pos), sig_y)
vanilla_exp.add_input('z_pos_initial', min(z_pos), max(z_pos), sig_z)
vanilla_exp.add_input('x_vel_initial', min(x_vel), max(x_vel), sig_vx)
vanilla_exp.add_input('y_vel_initial', min(y_vel), max(y_vel), sig_vy)
vanilla_exp.add_input('z_vel_initial', min(z_vel), max(z_vel), sig_vz)


# running MCMC chain 
vanilla_exp.run_chain(total=1000, burn=1000, every=2, n_chains=16, prior_only=True)
print(vanilla_exp.diagnostics_string())

prior_chains = vanilla_exp.get_chains(flattened=True)


# ### Diagnostics
# 
# $\hat{R}$ is a diagnostic that determines convergence (whether or not the chain has fully explored the whole distribution.) This value depends on the variance within chains and between chains. If this is too high it means that the chain has not been run long enough to fully converge to the target distribution.
# 
# #### Trace and Autocorrelation Plots for Each Input Parameter
# 
# Trace plots of samples versus the simulation index can be very useful in assessing convergence. The trace tells you if the chain has not yet converged to its stationary distribution—that is, if it needs a longer burn-in period. A trace can also tell you whether the chain is mixing well. A chain might have reached stationarity if the distribution of points is not changing as the chain progresses. The aspects of stationarity that are most recognizable from a trace plot are a relatively constant mean and variance. A chain that mixes well traverses its posterior space rapidly, and it can jump from one remote region of the posterior to another in relatively few steps.
# 
# From what we can observe from the trace plots produced, the results display a "perfect" trace plot. Note that the center of the chain appears to be around the midrange value of the parameter ranges, with very small fluctuations. This indicates that the chain *could* have reached the right distribution. The chain is mixing well; it is exploring the distribution by traversing to areas where its density is very low. You can conclude that the mixing is quite good here. 

# In[ ]:


def auto_trace(params):
    for param in params:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        vanilla_exp.trace_plot(param, ax=axes[0])
        vanilla_exp.autocorr_plot(param, ax=axes[1])
        axes[0].title.set_text("{} trace plot".format(param))
        axes[0].set(xlabel="iteration", ylabel="{} value".format(param))
        axes[1].title.set_text("{} autocorrelation plot".format(param))
        axes[1].set(xlabel="lag", ylabel="ACF")
        fig.tight_layout()
        fig.savefig(os.path.join(spec_root,"../06_surrogate_model/images/{}_trace_and_autocorrelation_plots.png".format(param)))
        plt.close(fig)


# In[ ]:


auto_trace(['x_pos_initial','y_pos_initial','z_pos_initial','x_vel_initial','y_vel_initial','z_vel_initial'])


# Before only the prior was considered, now the likelihood is calculated. We need to calculate the likelihood in order to generate likelihood plots discussed below. 

# In[ ]:


vanilla_exp.run_chain(total=1000, burn=1000, every=30, n_chains=16, prior_only=False)
print(vanilla_exp.diagnostics_string())

post_chains = vanilla_exp.get_chains(flattened=True)
post_points = np.stack(list(post_chains.values())).T


# ### Likelihood Plots
# 
# The likelihood is the probability that a particular outcome $x$ is observed when the true value of the parameter is $\theta$, equivalent to the probability mass on $x$; it is *not* a probability density over the parameter $\theta$.
# 
# The probability plot is a graphical technique for assessing whether or not a data set follows a given distribution such as the normal distribution.
# 
# The data are plotted against a theoretical distribution in such a way that the points should form approximately a straight line. Departures from this straight line indicate departures from the specified distribution. 

# In[ ]:


for key in post_chains.keys():
    fig, ax = plt.subplots(1, 1)
    ax.title.set_text(key)
    plots.likelihood_plot(ax, prior_chains[key], post_points=post_chains[key])
    fig.savefig(os.path.join(spec_root,"../06_surrogate_model/images/{}_likelihood_plot.png".format(key)))
    plt.close(fig)


# ### Appendix
# 
# #### Metropolis algorithm 
# 
# Provides a mechanism to explore the entire configuration space by random walk. At each step, the algorithm performs a random modification to the current state to obtain a new state. The new state is either accepted or rejected with a probability computed based on energy change in the transition. The states generated by the algorithm form a Markov chain.
# 
# |Metropolis Algorithm|
# |--------------------|
# |1: Randomize an input $g$.|
# |2: **repeat** |
# |3: &nbsp; Generate $g^*$ by performing a random trial move from $g$.|
# |4: &nbsp; Compute $Pr(g \rightarrow g^*)=$ min $\left\{1,\frac{Pr(g^*)}{Pr(g)}\right\}$|
# |5: &nbsp; if _random_ $(0,1]<Pr(g \rightarrow g^*)$ then $g \rightarrow g^*$.|
# |6: **until** equilibrium is attained.|
# 
# 
# #### Gibbs sampler 
# A special case of the Metropolis algorithm, which generates new states by using univariate conditional probabilities. Because direct sampling from the complex joint distribution of all random variables is difficult, Gibbs sampler instead simulate random variables one by one from the univariate conditional distribution. A univariate conditional distribution involves only one random variable conditioned on the rest variables having fixed values, which is usually in a simple mathematical form.
# 
# |Gibbs Sampling Algorithm|
# |------------------------|
# |1: Randomize an input $g$.|
# |2: **repeat** |
# |3: &nbsp; **for all** $i \in R$ **do**|
# |4: &nbsp;&nbsp;&nbsp;&nbsp; Compute $Pr(g_i = q \| g^i)$ for all $q \in Q$.|
# |5: &nbsp;&nbsp;&nbsp;&nbsp; Assign $g_i$ to the value $q$ with the probability $Pr(g_i = q \| g^i)$.|
# |6: &nbsp; **end for** |
# |7: **until** equilibrium is attained.|
