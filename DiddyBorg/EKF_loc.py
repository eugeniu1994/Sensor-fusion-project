import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
fig_size = (8,8)
from typing import Callable
import pandas as pd

def Extended_Kalman_filter(f: Callable[[np.ndarray, np.ndarray], np.ndarray], h: Callable[[np.ndarray], np.ndarray],
                           df_dx: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           dh_dx: Callable[[np.ndarray], np.ndarray],
                           x_init, cov_init, Q_discrete, R_discrete, y, u=None):
    length = y.shape[0]
    x = np.zeros((length, x_init.shape[0]))
    P = np.zeros((length, x_init.shape[0], x_init.shape[0]))
    x[0] = x_init
    P[0] = cov_init

    if u is None:
        u = np.zeros((length, 1))

    for i in range(length - 1):
        x_ = f(x[i], u[i])
        F_x = df_dx(x[i], u[i])
        P_ = F_x @ P[i] @ F_x.T + Q_discrete

        H_x = dh_dx(x_)
        S = H_x @ P_ @ H_x.T + R_discrete
        K_tran = sla.solve(S, H_x @ P_)
        P[i + 1] = P_ - K_tran.T @ S @ K_tran
        x[i + 1] = x_ + K_tran.T @ (y[i] - h(x_))
    return x, P


robot_init = np.array([0.5,0.5,np.pi/4])
robot_init = np.array([15.7, 47.5, np.deg2rad(90)])

'''t_robot = np.linspace(0., 5., 501)
dt_robot = t_robot[1] - t_robot[0]
u_robot = np.zeros((t_robot.shape[0], 2))

for i in range(t_robot.shape[0]):
    if 0 <= t_robot[i] < 1:
        u_robot[i, 0] = t_robot[i]
    elif 1 <= t_robot[i] < 4:
        u_robot[i, 0] = 1
    else:
        u_robot[i, 0] = 5 - t_robot[i]

    if 0 <= t_robot[i] < 2:
        u_robot[i, 1] = 0
    elif 2 <= t_robot[i] < 3:
        u_robot[i, 1] = np.pi
    else:
        u_robot[i, 1] = 0'''
imu = 'Datasets/data/task1/imu_reading_task1.csv'
IMU_static = pd.read_csv(imu, header=None)
bias_omega_z = np.mean(IMU_static.iloc[:,8])
print('bias_omega_z ',bias_omega_z)

csvFile = 'Datasets/data/task6/imu_tracking_task6.csv'
calib_data = np.loadtxt(csvFile, delimiter=',', skiprows=0)
t_robot = calib_data[:, 0]
GyZ = calib_data[:, 8]

w = np.array(GyZ).flatten()
print('w ', np.shape(w))
print('t_robot ', np.shape(t_robot))
u_robot = []
dt_robot = 0.06220197677612305
dt_robot /=2
vel = 5.5 #maxim velocity from task1 - part 4

for i in range(1,len(t_robot)):
    dt = t_robot[i] - t_robot[i-1]
    v,w_gyro = vel, w[i]-bias_omega_z  #m/s, degree/s
    w_gyro = np.deg2rad(w_gyro)
    #print('dt:{}, v:{}, w:{} '.format(dt,vel,w_gyro))
    inp = [v,w_gyro] #input velocity and gyroscope
    u_robot.append(inp)

u_robot = np.array(u_robot)
print('u_robot ', np.shape(u_robot))

Q_robot = np.diag([1e-2,1e-2,1e-2])
Q_robot_discrete = dt_robot*Q_robot
q_robot_discrete = np.random.multivariate_normal(np.zeros(3),Q_robot_discrete,(t_robot.shape[0]))
R_robot_discrete = 0.01*np.eye(2)
r_robot = np.random.multivariate_normal(np.zeros(2),R_robot_discrete,(t_robot.shape[0]))

def _f_robot(x: np.ndarray, u: np.ndarray):
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])
f_robot = np.vectorize(_f_robot, signature='(n),(m)->(n)')

def f_robot_discrete(x: np.ndarray, u: np.ndarray):
    return x + f_robot(x, u) * dt_robot

def df_dx_robot(x: np.ndarray, u: np.ndarray):
    jac = np.zeros((x.shape[0], x.shape[0]))
    jac[0, 2] = -u[0] * np.sin(x[2])
    jac[1, 2] = u[0] * np.cos(x[2])
    return jac

def df_dx_robot_discrete(x: np.ndarray, u: np.ndarray):
    return np.eye(x.shape[0]) + df_dx_robot(x, u) * dt_robot

#def _h_robot(x: np.ndarray):
#    return np.array([x[0], x[1]])
#h_robot = np.vectorize(_h_robot, signature='(n)->(2)')  ##Vectorize the stuff to avoid doing looping

def _h_robot(x:np.ndarray):
    return np.array([np.sqrt(x[0]*x[0]+x[1]*x[1]),np.arctan2(-x[1],x[0])-x[2]])

h_robot = np.vectorize(_h_robot, signature='(n)->(2)') ##Vectorize the stuff to avoid doing looping

#def dh_dx_robot(x: np.ndarray):
#    return np.array([[1, 0., 0.], [0., 1., 0.]])

def dh_dx_robot(x:np.ndarray):
    distance = np.sqrt(x[0]*x[0]+x[1]*x[1])
    return np.array([[x[0]/distance,x[1]/distance, 0.],
                     [-x[1]/distance,x[0]/distance, -1]])

def euler_maruyama_propagate(f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             x_init: np.ndarray, w: np.ndarray, dt: float, u: np.ndarray = None):
    length = w.shape[0]
    if u is None:
        u = np.zeros((length, 1))
    x_res = np.zeros((length, x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0] - 1):
        x_res[i + 1] = x_res[i] + f(x_res[i], u[i]) * dt + w[i]
    return x_res

x_robot = euler_maruyama_propagate(f_robot, robot_init, q_robot_discrete, dt_robot, u_robot)

y_robot = h_robot(x_robot)+r_robot
plt.figure()
plt.plot(y_robot[:,0],y_robot[:,1], 'ok', markersize=6, linewidth=0.5,alpha=0.25, label='measurements')
plt.legend()
plt.show()

x_robot_ekf_estimate,P_robot_ekf_estimate = Extended_Kalman_filter(f_robot_discrete,h_robot,df_dx_robot_discrete,dh_dx_robot,
                                               robot_init,Q_robot_discrete,Q_robot_discrete,
                                               R_robot_discrete,y_robot,u_robot)
stdev_robot_ekf = np.sqrt(np.diagonal(P_robot_ekf_estimate,axis1=1,axis2=2))

f, ax = plt.subplots(1,3,figsize=(2*fig_size[0],fig_size[1]))
for i in range(3):
    ax[i].plot(t_robot,x_robot_ekf_estimate[:,i],linewidth=0.5, color='red', label='$\hat x_{}$ EKF'.format(i+1))
    ax[i].fill_between(t_robot,x_robot_ekf_estimate[:,i]-2*stdev_robot_ekf[:,i],x_robot_ekf_estimate[:,i]+2*stdev_robot_ekf[:,i],label='conf. EKF \n $x_{}$'.format(i+1),color='r',alpha=0.2)
    ax[i].plot(t_robot,x_robot[:,i],'-k',linewidth=0.5, label='$x_{}$'.format(i+1))
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$x$')
    ax[i].legend()
plt.show()

f, ax = plt.subplots(1,2, figsize=(2*fig_size[0],2*fig_size[1]))
quiver_width=0.005
quiver_head_width=3.
quiver_head_length=4.
skip = 25

ax[0].plot(x_robot[:,0],x_robot[:,1], '-k', label='Robot-position-EM', linewidth=0.5)
ax[0].quiver(x_robot[::skip,0],x_robot[::skip,1],
               np.cos(x_robot[::skip,2]),np.sin(x_robot[::skip,2]),
               label='Direction-EM', linewidth=0.5, alpha=0.5, color='black',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[0].plot(x_robot_ekf_estimate[:,0],x_robot_ekf_estimate[:,1], '-r', label='Robot-position-EKF', linewidth=0.5)
ax[0].quiver(x_robot_ekf_estimate[::skip,0],x_robot_ekf_estimate[::skip,1],
               np.cos(x_robot_ekf_estimate[::skip,2]),np.sin(x_robot_ekf_estimate[::skip,2]),
               label='Direction-EKF', linewidth=0.5, alpha=0.5, color='red',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)

ax[0].plot(y_robot[:,0],y_robot[:,1], 'ok', markersize=6, label='Measurement', alpha=0.2)
ax[0].set_xlabel('$x_1$')
ax[0].set_ylabel('$x_2$')
ax[0].legend()

ax[1].plot(t_robot,x_robot[:,2]/np.pi, '-k', label='Orientation-EM', linewidth=0.5)
ax[1].plot(t_robot,x_robot_ekf_estimate[:,2]/np.pi, '-r', label='Orientation-EKF', linewidth=0.5)
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('$\phi / \pi$')
ax[1].legend()
plt.show()
