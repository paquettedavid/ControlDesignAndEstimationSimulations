import numpy as np
import matplotlib.pyplot as plt

def main():
    tf = 5 # final time
    Ts = 0.01 # sample rate
    t = np.arange(0,tf,Ts) # t0 to tf in increments of Ts
    n = np.size(t)
    ts = np.ones((n,1))

    # generate input step signal
    for i in range(0,n):
        if i > 50:
            if i > 450:
                ts[i] = 1
            elif i > 350:
                ts[i] = 2.5
            elif i > 250:
                ts[i] = 3
            elif i > 150:
                ts[i] = 2
            else:
                ts[i] = 1

    # dynamics definitions
    A_base = np.matrix('0.8277 0; 0 0.6896') # state transition matrix
    A = np.matrix('0.8277 0; 0 0.6896') # dynamics at operating point
    B = np.matrix('0.07289 0; 0 0.008352') # input gain matrix
    C = np.matrix('14.48 0.5919') # measurement gain matrix
    K = np.matrix('17.5  17.5') # feedback gain
    H = 0.5 # integral action gain
    G = 0.1 # nonlinear model gain
    base = 1 # DLO
    integrator = 0 # integral action
    x = np.matrix('0 ; 0') # state
    y = np.matrix('0') # output measurement

    U = np.matrix([[0], [0]])  # input
    R = np.matrix([[ts[0]],[0]]) # reference input

    x_sig = np.zeros((n,2))
    y_sig = np.zeros((n, 1))

    for k in range(0,n):
        #  reference tracking state feedback with integral action
        R = np.matrix([[ts[k][0]], [0]]) # grab new reference at sample k
        U = np.matrix([[H*integrator], [0]]) # set input to integrated error
        x = A*x + B*(U-K*x) # compute new state using state feedback
        y = C*x # compute output
        integrator = (R[0, 0] - y[0, 0]) + integrator  # integrate error using new output

        x_sig[k] = x.T
        y_sig[k] = y.T # "sample"/measure output from system

        # inject some, non-time varying, non-linear dynamics into our model when not at our DLO
        A = G*(y[0,0]-base) + A_base

        # gain schedule (using poly fit of tuned gains at several operating points)
        p = R[0, 0] - 1
        K = np.matrix([[1.05*pow(p,2)-4.35*p+17.5, -2.05*pow(p,2)-1.25*p+17.15]])
        # gain schedule (terrible piece-wise way)
        # if y[0,0] > 2.5:
        #    K = np.matrix('13 6.8')
        # elif y[0,0] > 1.5:
        #    K = np.matrix('14.2 14.2')
        # else:
        #    K = K_base

    plt.plot(t,ts) # plot sensor data
    plt.plot(t, y_sig)  # plot sensor data
    plt.show()

if __name__ == "__main__":
    main()