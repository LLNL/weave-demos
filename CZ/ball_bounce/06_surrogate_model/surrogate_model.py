import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

import sina.datastores.sql as sina_sql
import sina.utils
from sina.datastore import create_datastore
from sina.visualization import Visualizer

from trata import sampler as sm, composite_samples as cs, adaptive_sampler as ad
from ibis import mcmc as mc, plots
import statistics as sts
from sklearn.gaussian_process import GaussianProcessRegressor
import themis
import os
import sys

if sys.argv[1] == '-f':  # Not sure why this returns '-f' if no arguments are passed?
    spec_root = ''
    # use %matplotlib widget instead of %matplotlib notebook if using vscode, unless you install the matplotlib extension
    %matplotlib widget
else:
    spec_root = sys.argv[1]  # Modify path for Python script since this is running within its own step


# Baseline Initialization
database_baseline = os.path.join(spec_root, '../01_baseline_simulation/baseline/data/baseline_output.sqlite')
datastore_baseline = create_datastore(database_baseline)
recs_baseline = datastore_baseline.records
vis_baseline = Visualizer(datastore_baseline)

# Ensembles Initialization
database = os.path.join(spec_root, '../04_manage_data/data/ensembles_output.sqlite')
target_type = "csv_rec"
datastore = create_datastore(database)
recs = datastore.records
vis = Visualizer(datastore)


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
        fig.savefig("../06_surrogate_model/images/{}_trace_and_autocorrelation_plots.png".format(param))
        plt.close(fig)


if __name__ == "__main__":

    base_groups = set(x["group_id"]["value"] for x in recs_baseline.get_data(["group_id"]).values())

    # storing the simulation data locally to calculate descriptive statistics
    test_num_bounces, gravity, x_pos, y_pos, z_pos, x_vel, y_vel, z_vel = [], [], [], [], [], [], [], []

    # accessing and sampling the data for our surrogate model
    for group in base_groups:
        id_pool = list(recs_baseline.find_with_data(group_id=group))
        for _, rec_id in enumerate(id_pool):
            base_rec = recs_baseline.get(rec_id)
            data = base_rec.data_values
            test_num_bounces.append(data["num_bounces"])
            gravity.append(data["gravity"])
            x_pos.append(data["x_pos_initial"])
            y_pos.append(data["y_pos_initial"])
            z_pos.append(data["z_pos_initial"])
            x_vel.append(data["x_vel_initial"])
            y_vel.append(data["y_vel_initial"])
            z_vel.append(data["z_vel_initial"])

    # calculating sigma for our output
    nb_std = sts.stdev(test_num_bounces)

    # calculating sigmas for each input
    sig_g = sts.stdev(gravity)
    sig_x = sts.stdev(x_pos)
    sig_y = sts.stdev(y_pos)
    sig_z = sts.stdev(z_pos)
    sig_vx = sts.stdev(x_vel)
    sig_vy = sts.stdev(y_vel)
    sig_vz = sts.stdev(z_vel)


    groups = set(x["group_id"]["value"] for x in recs.get_data(["group_id"]).values())

    # instantiating the MCMC object
    vanilla_exp = mc.DefaultMCMC()

    # accessing and sampling the data for our surrogate model
    # we need to allocate training data from our previously run simulations to train a machine learning surrogate model
    # we will be training our model on gravity and num_bounces from our simulation data as well as position and velocity samples from Trata

    num_bounces_train, gravity_train, x_pos_train, y_pos_train, z_pos_train, x_vel_train, y_vel_train, z_vel_train = [], [], [], [], [], [], [], []

    for group in groups:
        id_pool = list(recs.find_with_data(group_id=group))
        for _, rec_id in enumerate(id_pool):
            rec = recs.get(rec_id)
            data = rec.data_values
            
            # our feature training data
            gravity_train.append(data["gravity"])
            x_pos_train.append(data["x_pos_initial"])
            y_pos_train.append(data["y_pos_initial"])
            z_pos_train.append(data["z_pos_initial"])
            x_vel_train.append(data["x_vel_initial"])
            y_vel_train.append(data["y_vel_initial"])
            z_vel_train.append(data["z_vel_initial"])

            # our target training data
            num_bounces_train.append(data["num_bounces"])


    # our sampled feature training data
    features_train = pd.DataFrame([gravity_train, x_pos_train, y_pos_train, z_pos_train, x_vel_train, y_vel_train, z_vel_train])
    features_train = features_train.transpose()
    num_bounces_train = pd.Series(num_bounces_train)

    # we have opted to use a Guassian Process Regressor as our surrogate model for this problem
    surrogate_model = GaussianProcessRegressor()

    # fitting the surrogate model using our parameter samples and the number of bounces simulated for those parameters
    surrogate_model.fit(features_train, num_bounces_train)        

    # adding the target testing data and our fitted model as output to the MCMC for each observed num_bounces
    for _ in num_bounces_test:
        vanilla_exp.add_output('output', 'num_bounces', surrogate_model, _, nb_std, 
                ['gravity', 'x_pos_initial', 'y_pos_initial','z_pos_initial','x_vel_initial','y_vel_initial','z_vel_initial'])

    # adding the input data to our MCMC
    vanilla_exp.add_input('gravity', min(gravity), max(gravity), sig_g)
    vanilla_exp.add_input('x_pos_initial', min(x_pos), max(x_pos), sig_x)
    vanilla_exp.add_input('y_pos_initial', min(y_pos), max(y_pos), sig_y)
    vanilla_exp.add_input('z_pos_initial', min(z_pos), max(z_pos), sig_z)
    vanilla_exp.add_input('x_vel_initial', min(x_vel), max(x_vel), sig_vx)
    vanilla_exp.add_input('y_vel_initial', min(y_vel), max(y_vel), sig_vy)
    vanilla_exp.add_input('z_vel_initial', min(z_vel), max(z_vel), sig_vz)


    # running MCMC chain 
    vanilla_exp.run_chain(total=10000, burn=10000, every=2, n_chains=16, prior_only=True)

    prior_chains = vanilla_exp.get_chains(flattened=True)
    prior_points = np.stack(prior_chains.values()).T

    auto_trace(['gravity','x_pos_initial','y_pos_initial','z_pos_initial','x_vel_initial','y_vel_initial','z_vel_initial'])

    # running MCMC chain with likelihood
    vanilla_exp.run_chain(total=10000, burn=10000, every=30, n_chains=16, prior_only=False)

    post_chains = vanilla_exp.get_chains(flattened=True)
    post_points = np.stack(post_chains.values()).T

    for key in post_chains.keys():
        fig, ax = plt.subplots(1, 1)
        ax.title.set_text(key)
        plots.likelihood_plot(ax, prior_chains[key], post_points=post_chains[key])
        fig.savefig("../06_surrogate_model/images/{}_likelihood_plot.png".format(key))
        plt.close(fig)
        