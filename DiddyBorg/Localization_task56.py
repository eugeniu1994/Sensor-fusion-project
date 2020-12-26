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

if __name__ == '__main__':
    camera_file = 'Dataset/data/task5/camera_localization_task5.csv'
    car = Car()
    car.localize(csvFile=camera_file)

    camera_file = 'Datasets/data/task6/camera_tracking_task6.csv'
    imu_file = 'Datasets/data/task6/imu_tracking_task6.csv'
    motor_file = 'Datasets/data/task6/motor_control_tracking_task6.csv'

    car.Estimate_pose_based_on_camera(cam_csv=camera_file)
