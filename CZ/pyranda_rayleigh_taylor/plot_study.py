import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import kosh
import os
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.model_selection import LeaveOneOut
import argparse

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--store", help="path to kosh/Sina store")
p.add_argument("--name", help="name for the ensembe of datasets plotted", required=True)
p.add_argument("--run-type", default="sim", help="run type to search for")
p.add_argument("--nmodels", type=int, help="number of times to train models")

args = p.parse_args()

rng = np.random.default_rng()

# Version 2 of sim/exp (making thickness the same at t0 for the 2)
#  Note on velocity-thickness
#dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240814-074047/run-pyranda/*"  # Exp-data (128/8)
store = kosh.connect(args.store)

# Create an ensemble or use an existing one
try:
    ensemble = next(store.find_ensembles(name=args.name))
except Exception:
    # does not exist yet
    ensemble = store.create_ensemble(name=args.name)

# Get NT times at which the layer mixing widths hit NT % of max
# We will train a models at a certain points in time
NTpts = args.nmodels
Tmin = 0.0
Tmax = 60.0  # maybe even 70?
samples = []  # atwood, vel, t0,... tN

sample_times = np.linspace(Tmin,Tmax,NTpts)
if len(sample_times) == 1:
    sample_times = [Tmax]


N_cases = len(list(store.find(types="pyranda", run_type=args.run_type, ids_only=True)))
for i, case in enumerate(store.find(types="pyranda", run_type=args.run_type), start=1):
    # Let's add this dataset to our ensemble
    #print("******************************")
    #print("DS:", case.id)
    #print("******************************")
    ensemble.add(case)

    # Le'ts retrieve var of interest
    time  = case["variables/time"][:] # time
    width = case["variables/mixing width"][:]  # Width
    mixed = case["variables/mixedness"][:]  # Mixedness
    atwood   = case.atwood_number
    velocity = case.velocity_magnitude
    lbl = f"Vel: {velocity} - At: {atwood}"
    plt.figure(2)
    plt.plot( time, width, '-o', label=lbl)
    for st in sample_times:
        plt.axvline(x=st, color='b', label=f"{st} s")
    plt.xlabel("Time")
    plt.ylabel("Mix Width")
    plt.title("Rayleigh-Taylor Simulations")
    if i == N_cases:
        fnm = "all_mixing_width.png"
        plt.savefig(fnm)
        ensemble.associate(fnm, "png", metadata={"title":lbl}) 

    # Plotting to show the input sampling design
    plt.figure(1)
    plt.plot(atwood, velocity, 'ko')
    plt.xlabel("Atwood number")
    plt.ylabel("Velocity magnitude")
    plt.title("Latin Hypercube Space-Filling Design")
    if i == N_cases:
        fnm = "atwood_vs_vel.png"
        plt.savefig(fnm)
        ensemble.associate(fnm, "png", metadata={"title":'atwood vs velocity'}) 


    # For each time, qoi, get NTpts
    #  Sample = [atwood, velocity, w(0), w(1), w(2) ...]
    sample_widths = np.interp(sample_times, time, width)
    sample = np.insert( sample_widths, 0, atwood)
    sample = np.insert( sample, 1, velocity)

    samples.append( sample )

samples = np.array( samples )

# Save for next step
header = f"# 'atwood' 'velocity' "
for ii in range(NTpts):
    header += " 'width-%s' " % ii
fnm = f"rt_{args.run_type}_data.csv"
np.savetxt(fnm, samples, delimiter=',',header=header)
#associate with ensemble
ensemble.associate(fnm, "pandas/csv", metadata={"gp_data":True})


############################
#   Fitting GP Models
############################

from matplotlib.colors import Normalize

xgp = samples[:,0:2]  # Get inputs
scaler = MMS()
scaled_samples = scaler.fit_transform(xgp)

# Get inputs for 2D plots
atwoods    = np.linspace(.25,.75, 100)
velocities = np.linspace(.75, 1.25, 100)
at2d, vel2d = np.meshgrid(atwoods, velocities)
atwoods = at2d.flatten().reshape(-1,1)
velocities = vel2d.flatten().reshape(-1,1)
inputs = np.concatenate( (atwoods, velocities), axis=1 )
scaled_inputs = scaler.transform(inputs)

GP_times = []
# Fitting a GP model for NTpts in time
for ii in range(NTpts):
    sample_time = sample_times[ii]
    y = samples[:,2+ii]  # Get width at this time-slice
    GP_times.append(GPR().fit(scaled_samples, y))

    # See GP prediction in 2D
    pred, std = GP_times[ii].predict(scaled_inputs, return_std=True)

    fig_num = 3 + ii
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    pred2d = pred.reshape(at2d.shape)
    std2d = std.reshape(at2d.shape)
    mycol = cm.jet((std2d - std.min()) / (std.max() - std.min()))
    cmap = plt.colormaps["jet"]
    plot = ax.plot_surface(at2d, vel2d, pred2d, facecolors=mycol)
    fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, label="Standard Error")
    #fig.colorbar(plot)
    #ax.contourf(at2d, vel2d, std2d, zdir='z', offset=0, cmap='coolwarm')
    ax.set_xlabel('Atwood')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Width')
    plt.title(f"GP Model at {sample_times[ii]} s")
    fnm = f"GP_at_{sample_time}_s.png"
    ax.figure.savefig(fnm)
    ensemble.associate(fnm, "png", metadata={"title":'2D GP'})

# Leave-one-out cross validation
loo = LeaveOneOut()

outputs = samples[:, 2:]

time_point_error = []
for ii in range(NTpts):
    loo_pred = []
    loo_bar = []
    loo_sqerror = []
    for i, (train_index, test_index) in enumerate(loo.split(scaled_samples)):
        y = outputs[:, ii]
        gp_model = GPR().fit(scaled_samples[train_index, :], y[train_index])
        pred, std = gp_model.predict(scaled_samples[test_index, :], return_std=True)
        loo_pred.append(pred)
        loo_bar.append((pred + std * 1.96) - (pred - std * 1.96))
        loo_sqerror.append((y[test_index] - pred)**2)
    plt.figure(3 + NTpts + ii)
    plt.errorbar(outputs[:, ii].flatten(),
                 np.array(loo_pred).flatten(),
                 yerr=np.array(loo_bar).flatten(),
                 fmt='o',
                 label='GP')
    plt.plot([2,7], [2,7], 'r-', label='Exact')
    plt.xlabel("Actual Mix Width")
    plt.ylabel("Predicted Mix Width")
    plt.title(f"Leave-One_Out Cross Validation {sample_times[ii]} s")
    plt.legend()
    fnm = f"LOO_xv_at_{sample_times[ii]}_s.png"
    plt.savefig(fnm)
    ensemble.associate(fnm, "png", metadata={"title":'LOO cross val'})

plt.pause(.1)
