import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kosh
import os
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
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
    print("******************************")
    print("DS:", case.id)
    print("******************************")
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
    ax.plot_surface(at2d, vel2d, pred2d)
    ax.contourf(at2d, vel2d, std2d, zdir='z', offset=0, cmap='coolwarm')
    ax.set_xlabel('Atwood')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Width')
    fnm = f"GP_at_{sample_time}_s.png"
    ax.figure.savefig(fnm)
    ensemble.associate(fnm, "png", metadata={"title":'2D GP'})

plt.pause(.1)
