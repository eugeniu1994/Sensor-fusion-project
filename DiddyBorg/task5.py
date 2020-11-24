import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sensor_fusion as sf
import robot_n_measurement_functions as rnmf
import lsqSolve as lsqS

camera_file = 'Datasets/data/task5/camera_localization_task5.csv'
# time, qr number, Cx, Cy, w,h, di, phi
Camera = sf.Sensor('Camera', sf.CAMERA_COLUMNS, meas_record_file=camera_file, is_linear=False, start_index=0)

x_true = np.array([61.0, 33.9, 0.0])
x_true = np.array([61.0, 33.9, 90.0])

x_init = np.array([0, 0, 0])  # x_true + np.random.randn(3)

R_one_diag = np.array([2, 10])

# params = {'x_sensors':np.zeros((1,2))}
I_max = 10
gamma = 1
params_LSQ = {'x_sensors': None,
              'R': None,
              'LR': None,
              # cholesky factorization of a matrix (chol(a) in matlab returns an upper triangular matrix, but linalg.cholesky(a) returns a lower triangular matrix)
              'Rinv': None,
              'gamma': gamma,
              'I_max': I_max,
              'Line_search': False,
              'Line_search_n_points': 10,
              'Jwls': lsqS.Jwls
              }

# %%
Camera.reset_sampling_index()
Camera.get_measurement()
# %%
y_raw = Camera.get_measurement()
print('y_raw ', np.shape(y_raw))
weight = y_raw[:, 3]
height = y_raw[:, 4]
c_x = y_raw[:, 1]

dist = rnmf.QRCODE_SIDE_LENGTH * rnmf.PERCEIVED_FOCAL_LENGTH / height
direct = np.arctan2(c_x, rnmf.PERCEIVED_FOCAL_LENGTH)  # + x_true[-1]
angle_qr = np.arccos(np.minimum(weight, height) / height)

corrected_dist = dist / np.cos(direct) + 0.5 * rnmf.QRCODE_SIDE_LENGTH * np.sin(angle_qr)
print('corrected_dist ', np.shape(corrected_dist))
y_raw[:, 5] = corrected_dist  # dist/np.cos(direct)
# y_raw[:,5] = dist
direct -= x_true[-1]
y_raw[:, 6] = direct

n_qr_codes = y_raw.shape[0]
print('n_qr_codes ', n_qr_codes)
# time, qr number, Cx, Cy, w,h, di, phi

y = y_raw[:, 5:].flatten()
print('y ', np.shape(y))

qr_pos = rnmf.QRCODE_LOCATIONS[y_raw[:, 0].astype('int'), 1:]
params_LSQ['x_sensors'] = qr_pos

R = np.diag(np.kron(np.ones(n_qr_codes), R_one_diag))
params_LSQ['R'] = R
params_LSQ['LR'] = np.linalg.cholesky(R).T
params_LSQ['Rinv'] = np.diag(1 / np.diag(R))
xhat_history_GN, J_history_GN = \
    lsqS.lsqsolve(y=y, g=rnmf.h_cam, G=rnmf.H_cam, x_init=x_init, params=params_LSQ,
                  method='gauss-newton')


x_ls = xhat_history_GN
best_X = x_init
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

axs[0, 1].plot(x_ls[2, :]* rnmf.RAD_TO_DEG, label='Orientation:{}'.format(np.round(best_X[2], 2)))
axs[0, 1].set_title('Orientation')
axs[0, 1].set(xlabel='time', ylabel='Orientation')
axs[0, 1].grid(True)
axs[0, 1].legend()

plt.show()







fig, ax = plt.subplots(4, 1)
ax[0].plot(xhat_history_GN[0, :], label='Px')
ax[1].plot(xhat_history_GN[1, :], label='Py')
ax[2].plot(xhat_history_GN[2, :] * rnmf.RAD_TO_DEG, label='Psi')
# ax[2].plot(xhat_history_GN[2,:] , label='Psi')

# ax[2].plot(xhat_history_GN[2,:], label='Psi')
plt.legend()
# plt.show()
x_init = xhat_history_GN[:, -1]
x_init[-1] = np.rad2deg(x_init[-1])
print('x_init ', x_init)
x_init = np.around(x_init, decimals=2)
print('np.radians(90) ', np.radians(90))
print('x_init ', x_init)
print('x_true ', x_true)

# ax[2].plot(xhat_history_GN[2,:]*rnmf.RAD_TO_DEG , label='Psi')
# ax[3].quiver(xhat_history_GN[0,:],xhat_history_GN[1,:],
#               np.cos(np.deg2rad(xhat_history_GN[2,:])),np.sin(np.deg2rad(xhat_history_GN[2,:])),
#               linewidth=0.5, alpha=0.5, color='black')

theta = xhat_history_GN[2, -1]
theta = np.deg2rad(theta)
print('theta = ', xhat_history_GN[2, -1])

ax[3].quiver(xhat_history_GN[0, -1], xhat_history_GN[1, -1],
             np.cos(theta), np.sin(theta),
             linewidth=0.6, alpha=1.0, color='red')
plt.show()

