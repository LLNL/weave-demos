import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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
NTpts = 11
Tmin = 0.0
Tmax = 60.0  # maybe even 70?
samples = []  # atwood, vel, t0,... tN

sample_times = np.linspace(Tmin,Tmax,NTpts)
print("SAMPLE SPACE SHAPE:", NTpts, sample_times.shape)


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
    plt.figure(3)
    plt.plot( time, width, '-o', label=lbl)
    if i == N_cases:
        fnm = "all_mixing_width.png"
        plt.savefig(fnm)
        ensemble.associate(fnm, "png", metadata={"title":lbl}) 


    if i == N_cases:
        print("Atwood: %s  Vel: %s" % (atwood,velocity))
        plt.figure(4)
        plt.plot(time, width, '-o', label=lbl)
        plt.legend()
        fnm = f"atwood_{atwood}-vel{velocity}_mixing_width.png"
        plt.savefig(fnm)
        # Associate plot with ensemble
        #ensemble.associate(fnm, "png", metadata={"title":lbl}) 

    plt.figure(1)
    plt.plot(atwood,velocity,'ko')
    if i == N_cases:
        fnm = "atwood_vs_vel.png"
        plt.savefig(fnm)
        ensemble.associate(fnm, "png", metadata={"title":'atwood vs velocity'}) 

    print( width.max(), time.max() )

    # For each time,qoi, get NTpts
    #  Sample = [atwood, velocity, w(0), w(1), w(2) ...]
    sample_widths = np.interp(sample_times,time,width)
    sample = np.insert( sample_widths, 0, atwood)
    sample = np.insert( sample, 1, velocity)
    samples.append( sample )

plt.pause(.1)

samples = np.array( samples )
header = f"# 'atwood' 'velocity' "
for ii in range(NTpts):
    header += " 'width-%s' " % ii

# Save for next step
fnm = f"rt_{args.run_type}_data.csv"
np.savetxt(fnm, samples, delimiter=',',header=header)
#associate with ensemble
ensemble.associate(fnm, "pandas/csv", metadata={"gp_data":True})


def getGP(x,y):
    #scaler = MMS()
    #scaled_samples = scaler.fit_transform(x)
    #surrogate_model = GPR().fit(scaled_samples, y)
    surrogate_model = GPR().fit(x, y)
    return surrogate_model #,scaler



xgp = samples[:,0:2]  # Get inputs
GP_times = []
for ii in range(NTpts):
    sample_time = sample_times[ii]
    y = samples[:,2+ii]  # Get width at this time-slice
    GP_times.append(  getGP(xgp,y)   )

predicted_widths = []
predicted_std    = []
predict_inputs  = np.array([atwood,velocity]).reshape(-1, 1).T
for ii in range(NTpts):
    pred, std = GP_times[ii].predict( predict_inputs, return_std=True)
    predicted_widths.append( pred )
    predicted_std.append( std )

predicted_widths = np.array(predicted_widths)
predicted_std = np.array(predicted_std)
plt.figure(4)
plt.plot(sample_times,predicted_widths,'ko',label="GP prediction")
plt.legend()
plt.pause(.1)
#breakpoint()

# samples = np.array( samples )


scaler = MMS()
scaled_samples = scaler.fit_transform(samples[:,:3])

surrogate_model = GPR().fit(scaled_samples, samples[:,3])

time = time.reshape(-1,1)
atw = np.zeros(time.shape) + atwood
vel = np.zeros(time.shape) + velocity
inputs = np.concatenate((atw, vel, time), axis=1)
scaled_inputs = scaler.transform(inputs)

plt.figure(1)
plt.plot(atwood,velocity,'gx')


prediction, std = surrogate_model.predict(scaled_inputs, return_std=True)
plt.figure(2)
plt.plot(time, prediction, 'g-')
plt.plot(time, prediction + 1.96 * std, 'g--')
plt.plot(time, prediction - 1.96 * std, 'g--')

atwpt = (.65-.3) * rng.random() + .3
velpt = (1.15-.85) * rng.random() + .85

plt.figure(1)
plt.plot(atwpt,velpt,'rx')

plot_time = np.linspace(0.0, 84.0, 200).reshape(-1,1)
plot_atw = np.zeros(plot_time.shape) + atwpt
plot_vel = np.zeros(plot_time.shape) + velpt
plot_inputs = np.concatenate((plot_atw, plot_vel, plot_time), axis=1)
scaled_pinputs = scaler.transform(plot_inputs)

plot_pred, plot_std = surrogate_model.predict(scaled_pinputs, return_std=True)

plt.figure(2)
plt.plot(plot_time, plot_pred, 'k-')
plt.plot(plot_time, plot_pred + 1.96 * plot_std, 'b--')
plt.plot(plot_time, plot_pred - 1.96 * plot_std, 'b--')

plt.pause(.1)
