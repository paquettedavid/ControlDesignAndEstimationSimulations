import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    tf = 1000 # final time
    Ts = 1 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)

    # continuous dynamics definitions
    # first order plus dead time model
    Kp = 1 # process gain
    Tp = 30 # process time constant (seconds)
    Dp = 20 # process dead time (seconds)

    # FOPDT model to state space
    A = np.matrix([-1/Tp]) # dynamics at operating point
    B = np.matrix([Kp/Tp])  # input gain matrix
    C = np.matrix('1')  # measurement gain matrix

    # discrete dynamics (compute using continuous system)
    Ad = np.exp(A*Ts)
    Bd = (Ad-np.eye(np.shape(A)[0]))*B*np.linalg.inv(A)
    Cd = C

    # compute controller gains using cohen-coon tuning rules
    SM = 1 # stability margin 1 to 4, 1=less stable, 4= more stable
    Kc = 0.9/(SM*Kp)*(Tp/Dp+0.092)
    Ti = 3.33*Dp*((Tp+0.092*Dp)/(Tp+2.22*Dp))
    Td = 0

    # discrete controller gains
    Kcd = Kc
    Tid = (1/Ti)*Ts
    Tdd = Td/Ts

    # disturbance signals
    disturbance = np.zeros((n, 1))
    for i in range(0,round(n/4)): disturbance[i] = 0
    for i in range(round(n/4),n): disturbance[i] = 5
    stochastic_noise = np.random.normal(0, 0.01, size=n)  # additive white noise input signal

    # specify state, input and output signals
    x = np.matrix('0') # state, x(0)=1
    R = np.zeros((n, 1)) # reference
    # setpoint signal
    for i in range(0,800): R[i] = 20
    for i in range(800,n): R[i] = 35
    u_sig = np.zeros((n,1))
    y_sig = np.zeros((n, 1))
    integrator = 0
    pout = 0

    for k in range(0,n):
        # simulate feedback
        y = Cd*x  # compute output
        e = np.matrix([R[k]])-y # compute error
        integrator = integrator + e[0,0] # integrate error
        U = Kcd*(e + Tid*integrator + Tdd*(pout - y)) + disturbance[k] + stochastic_noise[k]
        pout = y
        x = Ad*x + Bd*U  # compute next state

        u_sig[k] = U.T
        y_sig[k] = y.T # "sample"/measure output from system


    plt.plot(t, y_sig, '-r', label='process output')
    plt.plot(t, u_sig,'-b', label='controller output')
    plt.plot(t, R, '-y', label='setpoint')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()