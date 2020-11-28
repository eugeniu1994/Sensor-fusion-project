import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import sensor_fusion as sf
import robot_n_measurement_functions as rnmf

np.random.seed(2020)
# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(5.0),  # variance of yaw angle
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation d_i,phi_i  covariance
#  Simulation parameter
INPUT_NOISE = np.diag([.4, np.deg2rad(5.0)]) ** 2
Measurement_Noise = np.diag([.3, .3]) ** 2

class getData:

    def __init__(self):
        self.index = 0
        camera_file = 'Datasets/data/task6/camera_tracking_task6.csv'
        imu_file = 'Datasets/data/task6/imu_tracking_task6.csv'
        IMU = pd.read_csv(imu_file, header=None)
        print('IMU ', np.shape(IMU))

        self.Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=camera_file, is_linear=False,
                                start_index=0)
        print('Camera ', self.Camera.meas_record.shape[0])
        x = np.zeros((self.Camera.meas_record.shape[0] // 3, 3), dtype=np.float)
        self.Camera.reset_sampling_index()
        i, vel = 0, 5.5
        vel /= 2.25
        landmarks, inputs = [], []
        while (self.Camera.current_sample_index < self.Camera.time.shape[0] and i < x.shape[0] - 1):
            i += 1
            y_ = self.Camera.get_measurement()
            if y_.shape[0] < 1:
                continue
            qr_row = y_[:, 0].astype('int')
            camera_time = self.Camera.current_time
            landmark = rnmf.QRCODE_LOCATIONS[qr_row, 1:]
            closest_index = IMU.iloc[:, 0].sub(camera_time).abs().idxmin()
            closest_u = np.array(IMU.iloc[closest_index, 8])
            # print('qr_row len:{}, qr_row:{}, time:{}, landmarks:{},closest_u:{}'.format(np.shape(qr_row),qr_row,camera_time, np.shape(landmark),closest_u))

            w_gyro = np.deg2rad(closest_u)
            u = [vel, w_gyro]  # steering command (vel, steering angle radians)
            inputs.append(u)
            landmarks.append(landmark)

        self.DT = .55
        print('landmarks:{},  inputs:{}'.format(np.shape(landmarks), np.shape(inputs)))

        self.landmarks = landmarks
        self.inputs = np.array(inputs)
        self.N = len(inputs)

    def getPoint(self):
        v = self.inputs[self.index, 0]
        yawrate = self.inputs[self.index, 1]

        u = np.array([[v], [yawrate]])
        l = self.landmarks[self.index]

        self.index += 1
        return u, l

data = getData()
DT = data.DT

def calc_input():
    u, _ = data.getPoint()
    return u

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0],
                  ])
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  ])

    x = F @ x + B @ u
    return x

def jacob_f(x, u):
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw)],
        [0.0, 0.0, 1.0*DT],
    ])

    return jF

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)
    z = observation_model(xTrue) + Measurement_Noise @ np.random.randn(2, 1)  # add noise to measurement

    ud = u + INPUT_NOISE @ np.random.randn(2, 1)  # add noise to input
    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud

def observation_model(x):
    land = data.landmarks[data.index]
    H = []
    for landmark_pos in land:
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = np.sqrt((px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2)

        Hi = np.array([
            [dist],
            [math.atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]
        ])
        H.append(Hi)
    H = np.mean(H, axis=0) # (2, 1)
    return H

def jacob_h(x):
    land = data.landmarks[data.index]
    jH = []
    for landmark_pos in land:
        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = np.sqrt(hyp)

        jH_i = np.array([
            [-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
            [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1*DT]
        ])
        jH.append(jH_i)

    jH = np.mean(jH, axis=0) # (2, 3)
    return jH

def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h(xPred)  # 2x3
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R

    K = np.linalg.solve(S,(PPred @ jH.T).T).T
    #K = PPred @ jH.T @ np.linalg.inv(S)

    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

    return xEst, PEst

def Estimate():
    x_init = np.array([15.7, 47.5, np.deg2rad(90)])
    xTrue = np.array([15.7, 47.5, np.deg2rad(90)]).reshape(-1, 1)
    xEst = xTrue
    PEst = np.eye(3)

    xDR = xTrue  # Dead reckoning

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', c='b', s=60) for landmark in data.landmarks]
    while data.index < data.N - 1:
        u = calc_input()
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

    skip = 10
    # plt.plot(hxDR[0, :].flatten(),
    #             hxDR[1, :].flatten(), "-k", label='D-R with noise ')

    plt.plot(hxEst[0, :].flatten(),
             hxEst[1, :].flatten(), "-r", label='EKF position')
    phi = hxEst[2, ::skip].flatten()
    plt.quiver(hxEst[0, ::skip].flatten(), hxEst[1, ::skip].flatten(),
               np.cos(phi), np.sin(phi),
               label='Direction-EKF', linewidth=0.1, alpha=0.8, color='green')

    phi = x_init[2]
    plt.quiver(x_init[0], x_init[1], np.cos(phi), np.sin(phi),
               linewidth=5., alpha=1.0, color='black', label='init-pose')

    plt.axis("equal")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    print('Main')
    Estimate()
