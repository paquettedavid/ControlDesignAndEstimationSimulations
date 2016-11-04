import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    tf = 10000 # final time
    Ts = 0.1 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)

    # continuous dynamics definitions
    # first order plus dead time model
    Kp = 4.16 # process gain
    Tp = 2000 # process time constant (seconds)
    Dp = 120 # process dead time (seconds)

    # FOPDT model to state space
    A = np.matrix([-1/Tp]) # dynamics at operating point
    B = np.matrix([Kp/Tp])  # input gain matrix
    C = np.matrix('1')  # measurement gain matrix

    # discrete dynamics (compute using continuous system)
    Ad = np.exp(A*Ts)
    Bd = (Ad-np.eye(np.shape(A)[0]))*B*np.linalg.inv(A)
    Cd = C

    # continuous controller gains
    Kc = 0.4
    Ti = 350
    Td = 0

    # discrete controller gains
    Kcd = Kc
    Tid = (1/Ti)*Ts
    Tdd = Td/Ts

    # specify state, input and output signals
    x = np.matrix('0') # state, x(0)=1
    R = np.matrix([[1]])  # reference
    u_sig = np.zeros((n,1))
    y_sig = np.zeros((n, 1))
    integrator = 0
    pout = 0

    for k in range(0,n):
        # simulate feedback
        y = Cd*x # compute output
        e = R-y # compute error
        integrator = integrator + e[0,0] # integrate error
        U = Kcd*( e + Tid*integrator + Tdd*(pout - y)) # compute controller output (input to system)
        pout = y
        x = Ad*x + Bd*U # compute next state

        u_sig[k] = U.T
        y_sig[k] = y.T # "sample"/measure output from system


    plt.plot(t, y_sig)  # plot output
    plt.plot(t, u_sig)  # plot controller effort
    plt.show()

if __name__ == "__main__":
    main()