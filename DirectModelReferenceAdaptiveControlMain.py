import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    tf = 70 # final time
    Ts = 0.01 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)

    # continuous plant with uncertain dynamics
    A = np.matrix('-1.5') # dynamics at operating point
    B = np.matrix('1.2')  # input gain matrix
    C = np.matrix('1')  # measurement gain matrix
    x = np.matrix('0')  # state, x(0)=1

    # continuous reference model
    A_ref = np.matrix('-1')
    B_ref = -1*A_ref # for unity DC gain across controller
    x_ref = np.matrix('0')

    # continuous model reference adaptive controller
    # u = k_x_hat*x + k_r_hat*r
    # adaptive parameters
    k_x_hat = np.matrix('0')
    k_r_hat = np.matrix('0')
    gamma_x = np.matrix('5')
    gamma_r = np.matrix('5')

    # discrete plant dynamics (computed using continuous system)
    Ad = np.exp(A*Ts)
    Bd = (Ad-np.eye(np.shape(A)[0]))*B*np.linalg.inv(A)
    Cd = C

    # compute discrete reference model
    A_ref_d = np.exp(A_ref * Ts)
    B_ref_d = (A_ref_d - np.eye(np.shape(A_ref)[0])) * B_ref * np.linalg.inv(A_ref)

    # data collection I/O signals
    u_sig = np.zeros((n,1))
    y_sig = np.zeros((n, 1))
    k_x_sig = np.zeros((n, 1))
    k_r_sig = np.zeros((n, 1))

    # define setpoint(s)
    r = np.concatenate((1*np.ones((1,1000)), 0*np.ones((1,1000)), -1*np.ones((1,1000)),0*np.ones((1,1000))), axis=1)
    r = np.concatenate((r,2*np.ones((1,1000))), axis=1)
    r = np.concatenate((r, 0 * np.ones((1, 1000))), axis=1)
    r = np.concatenate((r, 1 * np.ones((1, 1000))), axis=1)

    for k in range(0,n):

        #  compute controller output using estimated gains (input to our system)
        U = k_x_hat*x + k_r_hat*r[0,k] # feedback and feedforward controller

        # simulate uncertain(to the controller) plant
        x = Ad*x + Bd*U # compute next state
        y = Cd*x # compute output

        # update reference model
        x_ref = A_ref_d*x_ref + B_ref_d*r[0,k]

        # estimate gains using the following adaptive laws
        k_x_hat = np.matrix(-1)*gamma_x*x*(x - x_ref)*np.sign(B)*Ts + k_x_hat
        k_r_hat = np.matrix(-1)*gamma_r*r[0,k]*(x - x_ref)*np.sign(B)*Ts + k_r_hat

        # "sample"/measure output from system for plotting
        k_x_sig[k] = k_x_hat.T
        k_r_sig[k] = k_r_hat.T
        u_sig[k] = x_ref.T
        y_sig[k] = y.T


    plt.plot(t, y_sig, '-b', label= 'plant output')
    plt.plot(t, u_sig, '-r', label= 'desired track')
    plt.plot(t, k_x_sig, '-g', label='estimated k_x')
    plt.plot(t, k_r_sig, '-y', label='estimated k_r')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()