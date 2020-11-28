import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sensor_fusion as sf
import robot_n_measurement_functions as rnmf
import lsqSolve as lsqS
import matplotlib.patches as mpatches
import math
fig_size = (8,8)
from typing import Callable
import scipy.linalg as sla

from EKF import *

np.random.seed(2020)
class Car():
    def __init__(self):

        self.x_true = np.array([61.0, 33.9, 90.0])
        self.x_init = np.array([0,0,0])

        self.h0 = rnmf.QRCODE_SIDE_LENGTH #11.5 cm
        self.f = rnmf.PERCEIVED_FOCAL_LENGTH  #focal length pixel
        print('h0:{}, f:{}'.format(self.h0, self.f))

    def localize(self,csvFile = None):
        self.csvFile = csvFile
        self.Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=csvFile, is_linear=False, start_index=0)

        R_one_diag = np.array([2, 10])
        I_max,gamma,N = 20,1,10
        params_LSQ = {'x_sensors': None,
                      'R': None,
                      'LR': None,
                      'Rinv': None,
                      'gamma': gamma,
                      'I_max': I_max,
                      'Line_search': False,
                      'Line_search_n_points': N,
                      'Jwls': lsqS.Jwls}

        self.Camera.reset_sampling_index()
        self.Camera.get_measurement()

        y_raw = self.Camera.get_measurement()
        w = y_raw[:, 3]
        h = y_raw[:, 4]
        C_x = y_raw[:, 1]

        di = self.h0 * self.f / h       #according to equation 9
        phi = np.arctan2(C_x, self.f)  #according to equation 9

        angle_qr = np.arccos(np.minimum(w, h) / h)
        corrected_dist = di / np.cos(phi) + 0.5 * self.h0 * np.sin(angle_qr)


        y_raw[:, 5] = corrected_dist #update the di column
        phi -= self.x_true[-1] #update the angle
        y_raw[:, 6] = phi

        n_qr_codes = y_raw.shape[0]

        y_measurements = y_raw[:,5:].flatten() #take di and phi as measurements
        print('y_measurements ', np.shape(y_measurements))

        qr_pos = rnmf.QRCODE_LOCATIONS[y_raw[:, 0].astype('int'), 1:]
        params_LSQ['x_sensors'] = qr_pos
        R = np.diag(np.kron(np.ones(n_qr_codes), R_one_diag))
        params_LSQ['R'] = R
        params_LSQ['LR'] = np.linalg.cholesky(R).T
        params_LSQ['Rinv'] = np.diag(1 / np.diag(R))


        g = rnmf.h_cam #measurement model
        G = rnmf.H_cam #Jacobian of the measurement model
        method = 'gauss-newton'

        x_ls, J_ = lsqS.lsqsolve(y=y_measurements, g=g, G=G, x_init=self.x_init,
                                 params=params_LSQ,method=method)

        best_X = x_ls[:,-1].copy()
        best_X[-1] = np.rad2deg(best_X[-1])
        print('x_true ', self.x_true)
        print('best_X ',best_X)
        print('x_ls:{}'.format(np.shape(x_ls)))


        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs[0, 0].plot(x_ls[0, :], label='Px:{}'.format(np.round(best_X[0], 2)))
        axs[0, 0].set_title('X - position')
        axs[0, 0].set(xlabel='time', ylabel='X')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[1, 0].plot(x_ls[1, :], label='Py:{}'.format(np.round(best_X[1], 2)))
        axs[1, 0].set_title('Y - position')
        axs[1, 0].set(xlabel='time', ylabel='Y')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[0, 1].plot(x_ls[2, :] * rnmf.RAD_TO_DEG, label='Orientation:{}'.format(np.round(best_X[-1], 2)))
        axs[0, 1].set_title('Orientation')
        axs[0, 1].set(xlabel='time', ylabel='Orientation')
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim([-180, 180])
        axs[0, 1].legend()

        phi = np.deg2rad(best_X[-1])
        print('phi ',phi)


        axs[1, 1].quiver(best_X[0], best_X[1],np.cos(phi), np.sin(phi),
                     linewidth=2.9, alpha=1.0, color='red', label='estimated-pose')
        axs[1, 1].set_title('Final: X,Y and phi')
        axs[1, 1].set(xlabel='X', ylabel='Y')
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim([-5, 50])
        axs[1, 1].set_xlim([-5, 100])

        axs[1, 1].quiver(x_ls[0,:],x_ls[1,:],
                       np.cos(np.deg2rad(x_ls[2,:])),np.sin(np.deg2rad(x_ls[2,:])),
                       linewidth=0.5, alpha=0.5, color='green')

        axs[1, 1].quiver(self.x_init[0], self.x_init[1], np.cos(np.deg2rad(self.x_init[-1])),
                         np.sin(np.deg2rad(self.x_init[-1])),
                         linewidth=2.9, alpha=1.0, color='blue', label='init-pose')

        axs[1, 1].legend(loc='upper left')

        plt.show()

    def Estimate_pose_based_on_camera(self,cam_csv=None):
        self.Camera = sf.Sensor('Camera',sf.CAMERA_COLUMNS,meas_record_file=cam_csv,is_linear=False,start_index=0)
        self.x_true = np.array([15.7, 47.5, 90.0])
        self.x_init = self.x_true# np.array([0,0,0])
        x = np.zeros((self.Camera.meas_record.shape[0] // 3, 3), dtype=np.float)
        x[0, :] = self.x_init
        t = np.zeros(x.shape[0])
        t[0] = self.Camera.time[0]

        R_one_diag = np.array([2, 20])
        I_max,gamma,N = 10,1, 10
        params_LSQ = {'x_sensors': None,
                      'R': None,
                      'LR': None,
                      'Rinv': None,
                      'gamma': gamma,
                      'I_max': I_max,
                      'Line_search': False,
                      'Line_search_n_points': N,
                      'Jwls': lsqS.Jwls
                      }
        self.Camera.reset_sampling_index()
        i = 0
        while (self.Camera.current_sample_index < self.Camera.time.shape[0] and i < x.shape[0] - 1):
            i += 1
            y_ = self.Camera.get_measurement()
            n_qr_codes = y_.shape[0]

            if n_qr_codes < 3:
                x[i, :] = x[i - 1, :]
                continue

            w,h,c_x = y_[:, 3],y_[:, 4],y_[:, 1]

            di = self.h0 * self.f / h
            phi = np.arctan2(c_x, self.f)
            angle_qr = np.arccos(np.minimum(w, h) / h)

            corrected_dist = di / np.cos(phi) + 0.5 * self.h0 * np.sin(angle_qr)
            y_[:, 5] = corrected_dist
            y = y_[:, 5:].flatten()

            qr_row = y_[:, 0].astype('int')
            qr_pos = rnmf.QRCODE_LOCATIONS[qr_row, 1:]
            params_LSQ['x_sensors'] = qr_pos
            R = np.diag(np.kron(np.ones(n_qr_codes), R_one_diag))
            params_LSQ['R'] = R
            params_LSQ['LR'] = np.linalg.cholesky(R).T
            params_LSQ['Rinv'] = np.diag(1 / np.diag(R))
            xhat_history_GN, J_history_GN = lsqS.lsqsolve(y, rnmf.h_cam, rnmf.H_cam, x[i - 1, :], params_LSQ,
                                                          method='gauss-newton')
            x[i, :] = xhat_history_GN[:, -1]

        plt.figure()
        plt.title('Camera measurements & Jwls estimation')
        plt.plot(x[1:,0],x[1:,1],'-ok',linewidth=1,markersize=3, label='position')
        phi = x[1:,2]
        #plt.quiver(x[1:,0], x[1:,1], np.cos(phi), np.sin(phi),linewidth=2.9, alpha=1.0, color='red', label='Orientation')
        plt.quiver(x[1:, 0], x[1:, 1], -np.sin(phi), np.cos(phi),linewidth=2.9, alpha=1.0, color='red', label='Orientation')
        plt.grid(True)
        plt.quiver(x[-1,0], x[-1,1], -np.sin(phi[-1]), np.cos(phi[-1]),
                         linewidth=3., alpha=1.0, color='green', label='final-pose')

        plt.quiver(self.x_init[0], self.x_init[1], np.cos(np.deg2rad(self.x_init[-1])),
                         np.sin(np.deg2rad(self.x_init[-1])),
                         linewidth=3., alpha=1.0, color='blue', label='init-pose')
        plt.legend()
        plt.show()

    def Estimate_pose_Dead_reckoning(self, motor_file=None, cmd = True):
        def observation(x, u, DT):
            state = motion_model(x, u, DT)
            return state

        def motion_model(x, u, DT):
            F = np.array([[1.0, 0, 0, 0],
                          [0, 1.0, 0, 0],
                          [0, 0, 1.0, 0],
                          [0, 0, 0, 0]])

            B = np.array([[DT * math.cos(x[2, 0]), 0],
                          [DT * math.sin(x[2, 0]), 0],
                          [0.0, DT],
                          [1.0, 0.0]])

            return F @ x + B @ u

        #Timestamp, v,w
        motor = pd.read_csv(motor_file, header=None)
        print('motor ', np.shape(motor))

        # state  [x y yaw v]'
        x_dead = np.zeros((4, 1)) # Dead reckoning
        x_dead = np.array([[15.7], [47.5], [90], [0]]) # Dead reckoning

        print('x_dead ',np.shape(x_dead))
        x_dead_history = x_dead

        for i in range(1,len(motor)):
            t,v,w = motor.iloc[i,:]
            DT = t-motor.iloc[i-1,0]
            u = np.array([[v], [w]])
            x_dead  = observation(x_dead, u, DT)
            x_dead_history = np.hstack((x_dead_history, x_dead))

            plt.cla()
            x_pos = x_dead_history[0, :].flatten()
            y_pos = x_dead_history[1, :].flatten()
            plt.plot(x_pos, y_pos, label='XY - position')
            theta = x_dead_history[2, :].flatten()
            plt.quiver(x_pos, y_pos,
                             np.cos(theta), np.sin(theta),
                             linewidth=0.3, alpha=0.8, color='red', label = 'orientation')

            plt.quiver(x_pos[0], y_pos[0], np.cos(theta[0]),
                             np.sin(theta[0]),
                             linewidth=2, alpha=1.0, color='green', label='init-pose')
            plt.quiver(x_pos[-1], y_pos[-1], np.cos(theta[-1]),
                       np.sin(theta[-1]),
                       linewidth=2, alpha=1.0, color='blue', label='final-pose')
            plt.axis("equal")
            plt.title('dead-reckoning')
            plt.grid(True)
            plt.legend()

            plt.xlabel('X')
            plt.ylabel('Y')
            if cmd:
                plt.pause(0.01)
            if i==len(motor)-1:
                plt.show()

    def EKF_estimate(self,motor_file = None, camera_file = None):
        cam = pd.read_csv(camera_file, header=None)
        print('cam ', np.shape(cam))
        # Timestamp, v,w
        motor = pd.read_csv(motor_file, header=None)
        print('motor ', np.shape(motor))
        cam_time = cam.iloc[0,0],cam.iloc[-1,0]
        motor_time = motor.iloc[0, 0], motor.iloc[-1, 0]
        print('cam_time:{},  motor_time:{}'.format(cam_time,motor_time))

        motor = pd.read_csv(motor_file, header=None)
        print('motor ', np.shape(motor))
        '''
            -replace timesteps with int numbers autoincrement
            ---get data from camera first -> index of the detected qr codes
            ---based on current timestamp, get the clossest u from motor
            ---feed, landmarks pos and u to EKF
        '''
        self.Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=camera_file, is_linear=False, start_index=0)
        self.x_true = np.array([15.7, 47.5, 90.0])
        self.x_init = self.x_true  # np.array([0,0,0])
        x = np.zeros((self.Camera.meas_record.shape[0] // 3, 3), dtype=np.float)
        print('x shape is ', np.shape(x))
        self.Camera.reset_sampling_index()
        i = 0
        dt = 1
        dt=3.2
        dt = 6
        sigma_range = 0.05
        sigma_bearing = 0.05
        sigma_vel = 0.01
        sigma_steer = np.radians(1)

        ekf = RobotEKF(dt, wheelbase=0.12, sigma_vel=sigma_vel, sigma_steer=sigma_steer)
        ekf.x = array([[15.7, 47.5, np.deg2rad(90.0)]]).T
        ekf.P = np.diag([1, 1, 1])
        ekf.R = np.diag([sigma_range ** 2, sigma_bearing ** 2])

        while (self.Camera.current_sample_index < self.Camera.time.shape[0] and i < x.shape[0] - 1):
            i += 1
            y_ = self.Camera.get_measurement()
            if y_.shape[0] < 2:
                continue

            qr_row = y_[:, 0].astype('int')
            camera_time = self.Camera.current_time
            landmarks = rnmf.QRCODE_LOCATIONS[qr_row, 1:]
            closest_index = motor.iloc[:, 0].sub(camera_time).abs().idxmin()
            closest_u = np.array(motor.iloc[closest_index, 1:3])
            #print('qr_row len:{}, qr_row:{}, time:{}, landmarks:{},closest_u:{}'.format(np.shape(qr_row),qr_row,camera_time, np.shape(landmarks),closest_u))

            sim_pos = ekf.x.copy()  # simulated position
            u = closest_u  # steering command (vel, steering angle radians)
            #u[1] = np.deg2rad(-u[1])*3
            #u[0] *= 4
            u[1] = np.deg2rad(-u[1])
            #print('u ',u)

            plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)

            sim_pos = ekf.move(sim_pos, u, dt)  # simulate robot
            plt.plot(sim_pos[0], sim_pos[1], '-ok', linewidth=1, markersize=1)
            phi = sim_pos[-1]
            plt.quiver(sim_pos[0], sim_pos[1], np.cos(phi),
                       np.sin(phi),linewidth=.1, alpha=.5, color='red')

            ekf.predict(u=u)

            p_x, p_y = sim_pos[0, 0], sim_pos[1, 0]
            for lmark in landmarks:
                d = np.sqrt((lmark[0] - p_x) ** 2 + (lmark[1] - p_y) ** 2)
                a = atan2(lmark[1] - p_y, lmark[0] - p_x) - sim_pos[2, 0]
                z = np.array([[d + np.random.randn() * sigma_range], [a + np.random.randn() * sigma_bearing]])

                ekf.update(z, HJacobian=H_of, Hx=Hx, residual=residual,
                           args=(lmark), hx_args=(lmark))

        phi = np.deg2rad(self.x_init[-1])
        plt.quiver(self.x_init[0], self.x_init[1], np.cos(phi), np.sin(phi),
                   linewidth=2., alpha=1.0, color='blue', label='init-pose')

        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

    def EKF_estimate2(self, imu_file=None, camera_file=None):
        IMU = pd.read_csv(imu_file, header=None)
        print('IMU ',np.shape(IMU))
        t_robot = IMU.iloc[:, 0]
        GyZ = IMU.iloc[:, 8]
        w = np.array(GyZ).flatten()
        cam = pd.read_csv(camera_file, header=None)
        print('cam ', np.shape(cam))

        self.Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=camera_file, is_linear=False,start_index=0)
        self.x_true = np.array([15.7, 47.5, 90.0])
        self.x_init = self.x_true
        x = np.zeros((self.Camera.meas_record.shape[0] // 3, 3), dtype=np.float)
        print('x shape is ', np.shape(x))
        self.Camera.reset_sampling_index()
        i,vel = 0, 5.5
        landmarks,inputs = [],[]
        while (self.Camera.current_sample_index < self.Camera.time.shape[0] and i < x.shape[0] - 1):
            i += 1
            y_ = self.Camera.get_measurement()
            if y_.shape[0] < 2:
                continue

            qr_row = y_[:, 0].astype('int')
            camera_time = self.Camera.current_time
            landmark = rnmf.QRCODE_LOCATIONS[qr_row, 1:]
            closest_index = IMU.iloc[:, 0].sub(camera_time).abs().idxmin()
            closest_u = np.array(IMU.iloc[closest_index, 8])
            #print('qr_row len:{}, qr_row:{}, time:{}, landmarks:{},closest_u:{}'.format(np.shape(qr_row),qr_row,camera_time, np.shape(landmark),closest_u))

            u = [vel, closest_u]  # steering command (vel, steering angle radians)
            inputs.append(u)
            landmarks.append(landmark)

        print('landmarks:{},  inputs:{}'.format(np.shape(landmarks), np.shape(inputs)))
        [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', s=60) for landmark in landmarks]
        phi = np.deg2rad(self.x_init[-1])
        plt.quiver(self.x_init[0], self.x_init[1], np.cos(phi), np.sin(phi),
                   linewidth=2., alpha=1.0, color='black', label='init-pose')

        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    camera_file = 'Datasets/data/task5/camera_localization_task5.csv'
    car = Car()
    #car.localize(csvFile=camera_file)

    camera_file = 'Datasets/data/task6/camera_tracking_task6.csv'
    imu_file = 'Datasets/data/task6/imu_tracking_task6.csv'
    motor_file = 'Datasets/data/task6/motor_control_tracking_task6.csv'

    #car.Estimate_pose_based_on_camera(cam_csv=camera_file)
    #car.Estimate_pose_Dead_reckoning(motor_file = motor_file, cmd = True)

    car.EKF_estimate(motor_file=motor_file, camera_file=camera_file)
    #car.EKF_estimate2(imu_file=imu_file, camera_file=camera_file)
