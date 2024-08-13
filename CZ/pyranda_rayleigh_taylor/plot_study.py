import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

rng = np.random.default_rng()

# This is single mode cases (too easy for GP's)
# dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240813-113206/run-pyranda/*"
dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240813-125700/run-pyranda/*"
cases = glob.glob(dataDir)

samples = []

for case in cases:
    #if rng.random() < .85:
    #    continue
    dataFile = os.path.join(case,"RAYLEIGH_TAYLOR_2D.dat")
    data = np.loadtxt( dataFile )
    data = data[:,::10]
    time = data[0,:] # time
    qoi = data[1,:]  # Width
    #qoi = data[2,:]  # Mixedness
    plt.figure(3)
    plt.plot( data[0,:], qoi, '-o', label=os.path.basename(case) )
    #atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[0])
    #velocity = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[1])
    breakpoint()
    atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.SEED.')[0])
    velocity = float(os.path.basename(case).split('VEL.')[1])

    plt.figure(1)
    plt.plot(atwood,velocity,'ko')

    for ii in range(data.shape[1]):
        #sample = [atwood, velocity, data[0,ii], data[1,ii] ] 
        sample = [atwood, velocity, time[ii], qoi[ii] ]
        samples.append( sample )

samples = np.array( samples )

header = f"# 'atwood' 'velocity' 'time' 'mix_width'"
np.savetxt("rt_exp_data.csv", samples, delimiter=',')
scaler = MMS()
scaled_samples = scaler.fit_transform(samples[:,:3])

surrogate_model = GPR().fit(scaled_samples, samples[:,3])

time = data[0,:].reshape(-1,1)
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
