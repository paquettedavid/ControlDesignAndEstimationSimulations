import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    tf = 10 # final time
    Ts = 0.01 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)

    # continuous dynamics definitions
    A = np.matrix('-1') # dynamics at operating point
    B = np.matrix('1')  # input gain matrix
    C = np.matrix('1')  # measurement gain matrix
    # continuous LQR parameters
    Q = np.matrix('1') # state penalty
    R = np.matrix('1') # control penalty

    # discrete dynamics (computed using continuous system)
    Ad = np.exp(A*Ts)
    Bd = (Ad-np.eye(np.shape(A)[0]))*B*np.linalg.inv(A)
    Cd = C
    # discrete LQR parameters
    Qd = Q*Ts
    Rd = R/Ts

    # solve for optimal LQR gain
    P = sp.linalg.solve_discrete_are(Ad,Bd,Qd,Rd) # solve Riccati equation
    Klqr = np.linalg.inv(R)*np.transpose(B)*P # compute gain

    # specify state, input and output signals
    x = np.matrix('1') # state, x(0)=1
    R = np.matrix([[0]])  # input, regulate to 0 for all time
    u_sig = np.zeros((n,1))
    y_sig = np.zeros((n, 1))

    for k in range(0,n):
        #  optimal regulator using LQR state feedback
        U = (R-x)*Klqr # state feedback
        x = Ad*x + Bd*U # compute next state
        y = Cd*x # compute output

        u_sig[k] = U.T
        y_sig[k] = y.T # "sample"/measure output from system


    plt.plot(t, y_sig)  # plot output
    plt.plot(t, u_sig)  # plot controller effort
    plt.show()

if __name__ == "__main__":
    main()