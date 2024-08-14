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
#dataDir = "RT_STUDIES/rayleigh_taylor_pyranda_20240814-074047/run-pyranda/*"  # Exp-data (128/8)

cases = glob.glob(dataDir)

# Get NT times at which the layer mixing widths hit NT % of max
NTpts = 11
Tmin = 0.0
Tmax = 60.0  # maybe even 70?
samples = []  # atwood, vel, t0,... tN

sample_times = np.linspace(Tmin,Tmax,NTpts)


for case in cases:
    dataFile = os.path.join(case,"RAYLEIGH_TAYLOR_2D.dat")
    data = np.loadtxt( dataFile )
    time  = data[0,:] # time
    width = data[1,:]  # Width
    mixed = data[2,:]  # Mixedness
    plt.figure(3)
    plt.plot( data[0,:], width, '-o', label=os.path.basename(case) )

    try:
        # without seed
        atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[0])
        velocity = float(os.path.basename(case).split('ATWOOD.')[1].split('.VEL.')[1])
    except:
        # with seed 
        atwood   = float(os.path.basename(case).split('ATWOOD.')[1].split('.SEED.')[0])
        velocity = float(os.path.basename(case).split('VEL.')[1])

    if (case == cases[-1]):
        print("Atwood: %s  Vel: %s" % (atwood,velocity))
        plt.figure(4)
        plt.plot( data[0,:], width, '-o', label=os.path.basename(case) )
        plt.legend()

    plt.figure(1)
    plt.plot(atwood,velocity,'ko')

    print( width.max(), time.max() )

    # For each time,qoi, get NTpts
    #  Sample = [atwood, velocity, w(0), w(1), w(2) ...]
    sample_widths = np.interp(sample_times,time,width)
    sample = np.insert( sample_widths, 0, velocity)
    sample = np.insert( sample, 0, atwood)
    samples.append( sample )

plt.pause(.1)



samples = np.array( samples )



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

header = f"# 'atwood' 'velocity' "
for ii in range(NTpts):
    header += " 'width-%s' " % ii

#np.savetxt("rt_exp_data.csv", samples, delimiter=',',header=header)
np.savetxt("rt_sim_data.csv", samples, delimiter=',',header=header)

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
