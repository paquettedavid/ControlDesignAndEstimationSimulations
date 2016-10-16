import numpy as np
import matplotlib.pyplot as plt

def main():
    tf = 5
    t = np.arange(0,tf,0.01)
    n = np.size(t)
    ts = np.zeros((n,1)) #truth input signal

    # generate input step signal
    for i in range(0,n):
        if i > 100:
            ts[i] = 5

    noise_source_one = np.random.normal(0, 1, size=n) # additive white noise input signal
    noise_source_two = np.random.normal(0, 1, size=n)  # additive white noise input signal

    # continuous dynamics definition
    A = np.matrix('0.8277 0; 0 0.6896') # state transition matrix
    B = np.matrix('0.07289 0; 0 0.008352') # input gain matrix
    F = np.matrix('.008 ; .005') # system noise gain matrix
    W = np.matrix('.001') # measurement noise gain matrix
    C = np.matrix('14.48 0.5919') # measurement gain matrix
    x = np.matrix('0 ; 0') # state
    y = np.matrix('0') # output measurement
    w = noise_source_one # uniform distribution white noise signal
    v = noise_source_two # uniform distribution white noise signal
    x_sig_noise = np.zeros((n,2))
    y_sig_noise = np.zeros((n,1))
    K = np.matrix('0.0005') # kalman filter residual gain matrix
    Kf = np.matrix('-.054 -.099') # state feedback "half" matrix (explained lower)
    y_estimate_sig = np.zeros((n, 1))
    x_estimate_sig = np.zeros((n, 2))
    x_estimate = np.matrix('0 ; 0')
    u_control_output_sig = np.zeros((n, 1))
    U = np.matrix([[ts[0]],[0]])
    output_scale_factor = -1/0.00260
    # simulate system over time

    for k in range(0,n):
        # true system state evolution
        x = A*x + B*U + F*v[k]
        y = C*x + W*w[k]
        x_sig_noise[k] = x.T
        y_sig_noise[k] = y.T # "sample"/measure output from system

        # Kalman Filter state estimation
        # state prediction
        x_estimate = A * x_estimate + B * U
        # measurement prediction
        y_estimate = C * x_estimate
        # measurement residual (error)
        r_measurement = K * (y - y_estimate)
        # update state estimate
        x_estimate = x_estimate + r_measurement

        # use estimated state for full state feedback control.
        # (this is an odd example because half the input is a non-controllable disturbance)
        # and the other half is fed to half of a normal gain matrix.
        # I call it "half state feedback"
        U = np.matrix([Kf * x_estimate,ts[k]])

        # create time signals for plotting
        y_estimate_sig[k] = y_estimate.T
        x_estimate_sig[k] = x_estimate.T
        u_control_output_sig[k] = output_scale_factor*U[0]

    plt.plot(t,ts) # plot sensor data
    plt.plot(t, u_control_output_sig)  # plot sensor data

    plt.show()
if __name__ == "__main__":
    main()