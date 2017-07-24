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
    A = np.matrix('-1 0.5; 0 -1') # dynamics at operating point
    B = np.matrix('0;1')  # input gain matrix
    C = np.matrix('1 0; 0 1')  # measurement gain matrix
    x = np.matrix('0;0')  # state, x(0)=1
    L = np.matrix('0.75') # control uncertainty parameter
    Phi = np.matrix('1.3')
    # theta = abs(x(1))*x(1)

    # continuous reference model
    A_ref = np.matrix('-1 0; 0 -1')
    B_ref = np.matrix('0;1') # for unity DC gain across controller
    x_ref = np.matrix('0;0')
    r = np.matrix('1') # 'setpoint'

    # continuous model reference adaptive controller
    # adaptive parameters
    K_x_hat = np.matrix('0;0')
    K_r_hat = np.matrix('0')
    Phi_hat = np.matrix('0')
    Q = np.matrix('100 0; 0 100')
    G_x = np.matrix('5 0; 0 5')
    G_r = np.matrix('5')
    G_p = np.matrix('1')

    # discrete plant dynamics (computed using continuous system)
    Ad = sp.linalg.expm(A * Ts)
    Bd = np.linalg.pinv(A) * (Ad - np.eye(np.shape(A)[0])) * B
    Cd = C

    # discrete plant dynamics (computed using continuous system)
    A_ref_d = sp.linalg.expm(A_ref * Ts)
    B_ref_d = np.linalg.pinv(A_ref) * (A_ref_d - np.eye(np.shape(A_ref)[0])) * B_ref

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

        U = np.transpose(K_x_hat)*x + K_r_hat*r[0,k] - Phi_hat*abs(x[0,0]*x[0,0])

        # simulate uncertain(to the controller) plant
        x = Ad*x + Bd*L*(U + Phi*abs(x[0,0]*x[0,0]))# compute next state
        y = Cd*x # compute output

        # update reference model
        x_ref = A_ref_d*x_ref + B_ref_d*r[0,k]

        # estimate gains using the following adaptive laws
        P = sp.linalg.solve_discrete_lyapunov(A_ref_d, Q*Ts)
        K_x_hat = -G_x*x*np.transpose(x-x_ref)*P*B_ref_d*Ts + K_x_hat
        K_r_hat = -G_r*r[0,k]*np.transpose(x - x_ref)*P*B_ref_d*Ts + K_r_hat
        Phi_hat = G_p*abs(x[0,0]*x[0,0])*np.transpose(x - x_ref)*P*B*Ts + Phi_hat

        # "sample"/measure output from system for plotting
        k_x_sig[k] = K_x_hat.T[0,0]
        k_r_sig[k] = K_r_hat.T
        u_sig[k] = x_ref.T[0,1]
        y_sig[k] = x.T[0,1]


    plt.plot(t, y_sig, '-b', label= 'plant output')
    plt.plot(t, u_sig, '-r', label= 'desired track')
    plt.plot(t, k_x_sig, '-g', label='estimated k_x')
    plt.plot(t, k_r_sig, '-y', label='estimated k_r')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()