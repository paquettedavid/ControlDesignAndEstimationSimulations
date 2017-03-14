import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    tf = 2000 # final time
    Ts = 1 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)

    # process definition
    Kp = 1 # process gain
    Tp = 40 # process time constant (seconds)
    Dp = 25

    # process model definition
    Km = 1
    Tm = 40
    Dm = 25

    #constraints
    MVmax = 100
    MVmin = 0
    DMV = 5000 # max absolute MV speed
    disturbance = np.zeros((n,1))
    CV = np.zeros((n,1))
    internal_model = np.zeros((n,1))
    MV = np.zeros((n,1))
    setpoint = np.zeros((n, 1))

    # discrete process
    ap = np.exp(-Ts/Tp)
    bp = 1-ap

    # discrete process model
    am = np.exp(-Ts/Tm)
    bm = 1 - am

    # coincidence point (1 for first order)
    h = 1

    # closed loop time response
    cltr = 100

    # exponential reference trajectory
    reference_trajectory = 1-np.exp((-Ts*h*3)/cltr)

    # free mode
    free_mode = 1-pow(am,h)

    # disturbance signals
    for i in range(0,round(n/4)): disturbance[i] = 0
    for i in range(round(n/4),n): disturbance[i] = 5
    stochastic_noise = np.random.normal(0, .01, size=n)  # additive white noise input signal

    # setpoint signal
    for i in range(0,800): setpoint[i] = 20
    for i in range(800,n): setpoint[i] = 35

    for i in range(1,n):
        # "measure" physical process output
        CV[i] = ap*CV[i-1] + bp*Kp*(MV[i-1-Dp] + disturbance[i] + stochastic_noise[i])

        # update internal model
        internal_model[i] = am*internal_model[i-1] + bm*Km*MV[i-1-Dm]

        # predict process trajectory
        predicted = CV[i-Dm] + internal_model[i] - internal_model[i-Dm]

        # increment free mode
        # sfree = sm[i]*pow(am,h)-sm[i]
        d = (setpoint[i] - predicted)*reference_trajectory + internal_model[i]*free_mode

        # update base function
        sforced = free_mode*Km
        MV[i] = d/sforced

        # max/min speed constraints
        if MV[i] > MV[i-1]+DMV: MV[i] = MV[i-1] + DMV
        if MV[i] < MV[i-1]-DMV: MV[i] = MV[i - 1] - DMV
        # max/min value constraints
        if MV[i] > MVmax: MV[i] = MVmax
        if MV[i] < MVmin: MV[i] = MVmin


    plt.plot(t, CV, '-r', label='CV')  # plot process output
    plt.plot(t, MV,'-b', label='MV')  # plot controller effort
    plt.plot(t,internal_model, '-g',label='model') # plot model response
    plt.plot(t, setpoint, '-y', label='setpoint') #plot setpoint
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()