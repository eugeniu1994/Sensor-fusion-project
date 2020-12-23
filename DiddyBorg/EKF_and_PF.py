import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import sensor_fusion as sf
import robot_n_measurement_functions as rnmf
from numpy.random import randn, random, uniform
import scipy.stats as stast

np.random.seed(2020)
# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(5.0),  # variance of yaw angle
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation d_i,phi_i  covariance

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

class EKF_Localization():
    def __init__(self):
        self.data = getData()
        self.DT = self.data.DT

    def calc_input(self):
        u, _ = self.data.getPoint()
        return u

    def motion_model(self,x, u):
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0],
                      ])
        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT],
                      ])

        x = F @ x + B @ u
        return x

    def jacob_f(self,x, u):
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -self.DT * v * math.sin(yaw)],
            [0.0, 1.0, self.DT * v * math.cos(yaw)],
            [0.0, 0.0, 1.0*self.DT],
        ])

        return jF

    def observation(self, xTrue, xd, u):
        xTrue = self.motion_model(xTrue, u)
        z = self.observation_model(xTrue) + Measurement_Noise @ np.random.randn(2, 1)  # add noise to measurement

        ud = u + INPUT_NOISE @ np.random.randn(2, 1)  # add noise to input
        xd = self.motion_model(xd, ud)
        return xTrue, z, xd, ud

    def observation_model(self,x):
        land = self.data.landmarks[self.data.index]
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

    def jacob_h(self,x):
        land = self.data.landmarks[self.data.index]
        jH = []
        for landmark_pos in land:
            px = landmark_pos[0]
            py = landmark_pos[1]
            hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
            dist = np.sqrt(hyp)

            jH_i = np.array([
                [-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
                [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1*self.DT]
            ])
            jH.append(jH_i)

        jH = np.mean(jH, axis=0) # (2, 3)
        return jH

    def ekf_estimation(self,xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f(xEst, u)
        PPred = jF @ PEst @ jF.T + Q

        #  Update
        jH = self.jacob_h(xPred)  # 2x3
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + R

        K = np.linalg.solve(S,(PPred @ jH.T).T).T
        #K = PPred @ jH.T @ np.linalg.inv(S)

        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

        return xEst, PEst

def EstimateEkf():
    ekf = EKF_Localization()
    x_init = np.array([15.7, 47.5, np.deg2rad(90)])
    xTrue = np.array([15.7, 47.5, np.deg2rad(90)]).reshape(-1, 1)
    xEst = xTrue
    PEst = np.eye(3)

    xDR = xTrue  # Dead reckoning

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', c='b', s=60) for landmark in ekf.data.landmarks]
    while ekf.data.index < ekf.data.N - 1:
        u = ekf.calc_input()
        xTrue, z, xDR, ud = ekf.observation(xTrue, xDR, u)

        xEst, PEst = ekf.ekf_estimation(xEst, PEst, z, ud)
        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

    skip = 10
    # plt.plot(hxDR[0, :].flatten(),
    #             hxDR[1, :].flatten(), "-k", label='D-R with noise ')

    plt.title('Tracking with EKF')
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
    plt.legend()
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.xlim(-10, 130)
    #plt.ylim(-10, 130)
    plt.axis('equal')
    plt.show()

class RobotLocalizationParticleFilter():
    def __init__(self, N, x_dim, y_dim, landmarks, measure_std_error):
        self.data = getData()
        self.DT = .55# self.data.DT

        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.landmarks = landmarks
        self.R = measure_std_error

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1. / N)

        self.particles[:, 0] = uniform(0, x_dim, size=N)
        self.particles[:, 1] = uniform(0, y_dim, size=N)
        self.particles[:, 2] = uniform(0, 2 * np.pi, size=N)

    def setLandmarks(self, l):
        self.landmarks = l

    def create_uniform_particles(self, x_range, y_range, hdg_range):
        self.particles[:, 0] = uniform(x_range[0], x_range[1], size=self.N)
        self.particles[:, 1] = uniform(y_range[0], y_range[1], size=self.N)
        self.particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=self.N)
        self.particles[:, 2] %= 2 * np.pi

    def create_gaussian_particles(self, mean, var):
        self.particles[:, 0] = mean[0] + randn(self.N) * var[0]
        self.particles[:, 1] = mean[1] + randn(self.N) * var[1]
        self.particles[:, 2] = mean[2] + randn(self.N) * var[2]
        self.particles[:, 2] %= 2 * np.pi

    def predict(self, u, std, dt=1.):  # robot dynamics model
        self.particles[:, 2] += u[1] + randn(self.N) * std[1]
        self.particles[:, 2] %= 2 * np.pi

        d = u[0] * dt + randn(self.N) * std[0]
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * d
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * d

    def update(self, z):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            di = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1) #(N,)
            self.weights *= stast.norm(loc=di, scale=self.R).pdf(x=z[i])

        self.weights += 1.e-300
        self.weights /= sum(self.weights)  # normalize

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, random(self.N))

        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)  # normalize

    def resample_from_index(self, indexes):
        assert len(indexes) == self.N
        indexes = np.array(indexes, dtype=int)
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    def estimate(self):
        pos = self.particles[:, 0:3]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu) ** 2, weights=self.weights, axis=0)

        return mu, var

    def Gaussian(self, mu, sigma, x):
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        g = (np.exp(-((mu - x) ** 2) / (sigma ** 2) / 2.0) /
             np.sqrt(2.0 * np.pi * (sigma ** 2)))
        for i in range(len(g)):
            g[i] = max(g[i], 1.e-229)
        return g

    def systemic_resample(self, w):
        N = len(w)
        Q = np.cumsum(w)
        indexes = np.zeros(N)
        t = np.linspace(0, 1 - 1 / N, N) + random() / N

        i, j = 0, 0
        while i < N and j < N:
            while Q[j] < t[i]:
                j += 1
            indexes[i] = j
            i += 1

        return indexes

    def motion_model(self,x, u):
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0],
                      ])
        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT],
                      ])

        x = F @ x + B @ u
        return x

    def observation(self, xTrue, u):
        xTrue = self.motion_model(xTrue, u)
        #zs = None #self.observation_model(xTrue) + Measurement_Noise @ np.random.randn(2, 1)  # add noise to measurement
        zs = []
        for landmark in self.landmarks:
            d = np.sqrt((landmark[0] - xTrue[0,0]) ** 2 + (landmark[1] - xTrue[1, 0]) ** 2)
            zs.append(d + randn() * self.R) # (s, 1)
        zs = np.array(zs)

        return xTrue, zs

def EstimatePF():
    xTrue = np.array([15.7, 47.5, np.deg2rad(90)]).reshape(-1, 1)
    x_init = xTrue

    N, sensor_std_err = 5000, .1
    pf = RobotLocalizationParticleFilter(N, 130, 130, None, sensor_std_err)
    #pf.create_gaussian_particles(mean=x_init, var=(10, 10, np.pi / 4))
    pf.create_uniform_particles(x_range=(0, 130), y_range=(0, 130), hdg_range=(0, 6.28))
    xs = []

    plt.scatter(pf.particles[:, 0], pf.particles[:, 1], alpha=.1, c='green')
    [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', c='b', s=60) for landmark in pf.data.landmarks]
    phi = x_init[2]
    plt.quiver(x_init[0], x_init[1], np.cos(phi), np.sin(phi),
               linewidth=5., alpha=1.0, color='black', label='init-pose')
    plt.title('Tracking with PF')
    plt.grid(True)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    while pf.data.index < pf.data.N - 1:
        u, landmarks = pf.data.getPoint()
        pf.setLandmarks(l=landmarks)
        xTrue, zs = pf.observation(xTrue, u)

        pf.predict(u=u, std=(.05, np.deg2rad(5)), dt=pf.DT)
        pf.update(z=zs)

        pf.resample()
        #indexes = pf.systemic_resample(pf.weights)
        #pf.resample_from_index(indexes)

        mu, var = pf.estimate()
        xs.append(mu)

    skip = 10
    xs = np.array(xs)
    [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', c='b', s=60) for landmark in pf.data.landmarks]
    plt.plot(xs[:, 0], xs[:, 1], c='r', label='estimated position')

    phi = xs[::skip, 2].flatten()
    plt.quiver(xs[::skip, 0].flatten(), xs[::skip, 1].flatten(),
               np.cos(phi), np.sin(phi),
               label='Direction-PF', linewidth=0.1, alpha=0.8, color='green')

    phi = x_init[2]
    plt.quiver(x_init[0], x_init[1], np.cos(phi), np.sin(phi),
               linewidth=5., alpha=1.0, color='black', label='init-pose')
    plt.title('Tracking with PF')
    plt.grid(True)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.xlim(0, 130)
    #plt.ylim(0, 130)
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    print('Main')
    EstimatePF()
    EstimateEkf()

