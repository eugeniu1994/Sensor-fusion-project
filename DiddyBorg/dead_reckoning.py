import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

def gen_xy_from_yaw(yaw, dt, v0):
    psi0 = -np.pi / 2 + yaw[0]
    psiR = yaw
    dxdt = v0 * np.sin(psi0 - psiR)
    dydt = v0 * np.cos(psi0 - psiR)
    x = (dt * dxdt).cumsum()
    y = (dt * dydt).cumsum()
    return x, y

def plot_navigation(x0, y0, x1, y1, groundtruth, img=None):
    #x2, y2 = groundtruth.T
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    titles = ['GPS', 'Magnetometer', 'Google Maps']
    #axarr[0].plot(x0, y0, c='black')
    axarr[1].plot(x1, y1, c='black')
    #axarr[2].plot(x2, y2, c='orange', linestyle='dashed')
    plt.show()
    #axarr[2].imshow(img, aspect='auto')
    #for i in range(3):
    #    configure_axis(axarr[i], titles[i])
    #fig.tight_layout()
    #plt.savefig('../images/dead-reckoning.png')

def main(yaw, dt):
    vel = 6.4# 0.97           # Measured walking speed (m/s)
    #dt = 0.02            # Interval between yaw samples (s)
    dt = time[1] - time[0]

    x1, y1 = gen_xy_from_yaw(yaw, dt, vel)
    plt.plot(x1, y1, c='black', label='pose')
    plt.legend()
    plt.grid(True)
    plt.show()

import pandas as pd

if __name__ == '__main__':
    #Timestamp, a_x, a_y, a_z, roll, pitch in degree, g_x,g_y,g_z in degree/s,
    # m_x,m_y,m_z Gauss unit.
    csvFile = 'Datasets/data/task6/imu_tracking_task6.csv'
    #csvFile = 'Datasets/data/task2/imu_calibration_task2.csv'

    calib_data = np.loadtxt(csvFile, delimiter=',', skiprows=0)
    time = calib_data[:, 0]
    AcX, AcY, AcZ = calib_data[:, 1], calib_data[:, 2], calib_data[:, 3]
    roll, pitch = calib_data[:, 4], calib_data[:, 5]
    GyX, GyY, GyZ = calib_data[:, 6],calib_data[:, 7],calib_data[:, 8]
    MgX, MgY, MgZ =calib_data[:, 9],calib_data[:, 10],calib_data[:, 11]

    variable_names = ['time', 'roll', 'pitch', 'AcX', 'AcY', 'AcZ',
                      'GyX', 'GyY', 'GyZ', 'MgX', 'MgY', 'MgZ']
    plt.figure(figsize=(18, 15))
    for idx, name in enumerate(variable_names):
        exec(name + '=calib_data[:,' + str(idx) + ']')
        plt.subplot(4, 3, idx + 1)
        plt.plot(locals().get(name))
        if name in ['GyX', 'GyY', 'GyZ']:
            plt.ylim((-180, 180))
        plt.grid()
        plt.title(name, fontsize=14)


    plt.show()

    #------------------------------------------------------
    '''from pykalman import KalmanFilter

    # switch between two acceleration signals
    use_HP_signal = 1
    AcX = np.array(AcX).reshape(-1,1)

    print('AcX ', np.shape(AcX))
    if use_HP_signal:
        AccX_Value =  AcX# AccX_HP
        AccX_Variance = 0.0007
    else:
        AccX_Value = None# AccX_LP
        AccX_Variance = 0.0020

    # time step
    dt = 0.01

    # transition_matrix
    F = [[1, dt, 0.5 * dt ** 2],
         [0, 1, dt],
         [0, 0, 1]]

    # observation_matrix
    H = [0, 0, 1]

    # transition_covariance
    Q = [[0.2, 0, 0],
         [0, 0.1, 0],
         [0, 0, 10e-4]]

    # observation_covariance
    R = AccX_Variance

    # initial_state_mean
    X0 = [0,
          0,
          AccX_Value[0, 0]]

    # initial_state_covariance
    P0 = [[0, 0, 0],
          [0, 0, 0],
          [0, 0, AccX_Variance]]

    n_timesteps = AccX_Value.shape[0]
    n_dim_state = 3
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    kf = KalmanFilter(transition_matrices=F,
                      observation_matrices=H,
                      transition_covariance=Q,
                      observation_covariance=R,
                      initial_state_mean=X0,
                      initial_state_covariance=P0)

    # iterative estimation for each new measurement
    for t in range(n_timesteps):
        if t == 0:
            filtered_state_means[t] = X0
            filtered_state_covariances[t] = P0
        else:
            filtered_state_means[t], filtered_state_covariances[t] = (
                kf.filter_update(
                    filtered_state_means[t - 1],
                    filtered_state_covariances[t - 1],
                    AccX_Value[t, 0]
                )
            )

    f, axarr = plt.subplots(3, sharex=True)

    axarr[0].plot(AccX_Value, label="Input AccX")
    axarr[0].plot(filtered_state_means[:, 2], "r-", label="Estimated AccX")
    axarr[0].set_title('Acceleration X')
    axarr[0].grid()
    axarr[0].legend()
    axarr[0].set_ylim([-4, 4])

    #axarr[1].plot(Time, RefVelX, label="Reference VelX")
    axarr[1].plot(filtered_state_means[:, 1], "r-", label="Estimated VelX")
    axarr[1].set_title('Velocity X')
    axarr[1].grid()
    axarr[1].legend()
    axarr[1].set_ylim([-1, 20])

    #axarr[2].plot(Time, RefPosX, label="Reference PosX")
    axarr[2].plot(filtered_state_means[:, 0], "r-", label="Estimated PosX")
    axarr[2].set_title('Position X')
    axarr[2].grid()
    axarr[2].legend()
    axarr[2].set_ylim([-10, 1000])

    plt.show()'''
