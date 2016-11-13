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

    # embed control model for ramp tracking
    Ae = np.matrix([[0, 1, 0],[0, 0, C],[0, 0, A]])
    Be = np.matrix([[0],[0],[B]])
    Ce = np.matrix(np.shape(Ae)[0])
    # LQR "gains"
    Qe = np.matrix([[1, 0, 0],[0, 1, 0],[0, 0, 0]]) # state penalty
    Re = np.matrix('1') # control penalty

    # discrete dynamics (computed using embedded continuous system)
    Ad = sp.linalg.expm(Ae*Ts)
    if np.linalg.det(Ae) != 0: # if Ae is not singular
        Bd = (Ad - np.eye(np.shape(Ae)[0])) * Be * np.linalg.inv(Ae)
    else: # if singular, use pseudo inverse
        Bd = np.linalg.pinv(Ae)*(Ad-np.eye(np.shape(Ae)[0]))*Be
    Cd = Ce
    # discrete LQR parameters
    Qd = Qe*Ts
    Rd = Re/Ts

    # solve for optimal LQR gain
    P = sp.linalg.solve_discrete_are(Ad,Bd,Qd,Rd) # solve Riccati equation
    Klqr = np.linalg.inv(np.transpose(Bd)*P*Bd+Rd)*(np.transpose(Bd)*P*Ad)
    # specify state, input and output signals
    x = np.matrix([[0],[0],[0]]) # state, x(0)=1
    R = np.matrix([[0],[0],[1]])  # input, regulate to 0 for all time
    u_sig = np.zeros((n,1))
    y_sig = np.zeros((n, 1))

    for k in range(0,n):
        #  optimal regulator using LQR state feedback
        U = np.multiply((R-x),Klqr) # state feedback
        x = np.multiply(Ad,x) + np.multiply(Bd,U) # compute next state
        y = np.multiply(Cd,x) # compute output

        u_sig[k] = U[2,2].T
        y_sig[k] = y[2,2].T # "sample"/measure output from system

    plt.plot(t, y_sig)  # plot output
    plt.plot(t, u_sig)  # plot controller effort
    plt.show()

if __name__ == "__main__":
    main()