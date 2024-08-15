import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

rng = np.random.default_rng()

# Version 2 of sim/exp (making thickness the same at t0 for the 2)
#  Note on velocity-thickness
dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240814-072446/run-pyranda/*"  # Sim-data (32/2)
# dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240814-074047/run-pyranda/*"  # Exp-data (128/8)
# dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240815-104201/run-pyranda/*" # Exp-data (128/8 with diff means)

cases = glob.glob(dataDir)

# Get NT times at which the layer mixing widths hit NT % of max
# We will train a models at a certain points in time
NTpts = 1
Tmin = 0.0
Tmax = 60.0  # maybe even 70?
samples = []  # atwood, vel, t0,... tN

sample_times = np.linspace(Tmin,Tmax,NTpts)
if len(sample_times) == 1:
    sample_times = [Tmax]

for case in cases:
    dataFile = os.path.join(case,"RAYLEIGH_TAYLOR_2D.dat")
    data = np.loadtxt( dataFile )
    time  = data[0,:] # time
    width = data[1,:]  # Width
    mixed = data[2,:]  # Mixedness
    plt.figure(2)
    plt.plot(data[0,:], width, '-o', label=os.path.basename(case))
    for st in sample_times:
        plt.axvline(x=st, color='b', label=f"{st} s")
    plt.xlabel("Time")
    plt.ylabel("Mix Width")
    plt.title("Rayleigh-Taylor Simulations")

    try:
        # without seed
        atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[0])
        velocity = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[1])
    except:
        # with seed 
        atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.SEED.')[0])
        velocity = float(os.path.basename(case).split('VEL.')[1])

    # Plotting to show the input sampling design
    plt.figure(1)
    plt.plot(atwood, velocity, 'ko')
    plt.xlabel("Atwood number")
    plt.ylabel("Velocity magnitude")
    plt.title("Latin Hypercube Space-Filling Design")

    # For each time, qoi, get NTpts
    #  Sample = [atwood, velocity, w(0), w(1), w(2) ...]
    sample_widths = np.interp(sample_times, time, width)
    sample = np.insert( sample_widths, 0, velocity)
    sample = np.insert( sample, 0, atwood)
    samples.append( sample )

samples = np.array( samples )
# Save inputs and NTpts widths in csv for psuade
header = f"# 'atwood' 'velocity' "
for ii in range(NTpts):
    header += " 'width-%s' " % ii

# np.savetxt("rt_exp_data.csv", samples, delimiter=',',header=header)
# np.savetxt("rt_sim_data.csv", samples, delimiter=',',header=header)


############################
#   Fitting GP Models
############################

def getGP(x, y, scale=False):
    if scale:
        scaler = MMS()
        scaled_samples = scaler.fit_transform(x)
        surrogate_model = GPR().fit(scaled_samples, y)
    else:
        surrogate_model = GPR().fit(x, y)
    return [surrogate_model, scaler]

xgp = samples[:,0:2]  # Get inputs
scaler = MMS()
scaled_samples = scaler.fit_transform(xgp)

# Get inputs for 2D plots
atwoods    = np.linspace(.25,.75, 100)
velocities = np.linspace(.75, 1.25, 100)
at2d, vel2d = np.meshgrid(atwoods,velocities)
atwoods = at2d.flatten().reshape(-1,1)
velocities = vel2d.flatten().reshape(-1,1)
inputs = np.concatenate( (atwoods,velocities), axis=1 )
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

plt.pause(.1)
