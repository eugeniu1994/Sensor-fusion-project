import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2020)

def robot_dynamic(t, x, u):
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])

def robot_jacobian(t, x, u):
    jac = np.zeros((x.shape[0], x.shape[0]))
    jac[0, 2] = -u[0] * np.sin(x[2])
    jac[1, 2] = u[0] * np.cos(x[2])
    return jac

robot_init = np.array([15.7, 47.5, np.deg2rad(90)])

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

def euler(f, t_now, x_now, u_now, dt):
    return x_now + f(t_now, x_now, u_now) * dt

def euler_propagate(f, t, x_init, u, dt):
    x_res = np.zeros((t.shape[0], x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0] - 1):
        x_res[i + 1] = euler(f, t[i], x_res[i], u[i], dt)
    return x_res

def euler_stochatic_propagate(f, t, x_init, u, w, dt):
    x_res = np.zeros((u.shape[0], x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0] - 1):
        x_res[i + 1] = euler(f, t[i], x_res[i], u[i], dt) + w[i]
    return x_res

def linearization_propagate(f, jac, t, x_init, u, w, dt):
    x_res = np.zeros((t.shape[0], x_init.shape[0]))
    I = np.eye(x_init.shape[0])
    x_res[0] = x_init
    for i in range(x_res.shape[0] - 1):
        A = jac(t[i], x_res[i], u[i])
        F = (I + 0.5 * A * dt + A @ A * dt * dt / 6) * dt  # this is approximation
        x_res[i + 1] = x_res[i] + F @ f(t[i], x_res[i], u[i]) + F @ w[i]
    return x_res

def rk4(f, t_now, x_now, u_now, dt):
    k1 = f(t_now, x_now, u_now)
    k2 = f(t_now + 0.5 * dt, x_now + 0.5 * dt * k1, u_now)
    k3 = f(t_now + 0.5 * dt, x_now + 0.5 * dt * k2, u_now)
    k4 = f(t_now + dt, x_now + dt * k3, u_now)
    return x_now + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def rk4_propagate(f, t, x_init, u, dt):
    x_res = np.zeros((u.shape[0], x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0] - 1):
        x_res[i + 1] = rk4(f, t[i], x_res[i], u[i], dt)
    return x_res

Q_robot = np.diag([1e-2, 1e-2, 1e-2])
Q_robot_discrete = dt_robot * Q_robot
q_robot = np.random.randn(t_robot.shape[0], 3) @ np.linalg.cholesky(Q_robot)
q_robot_discrete = np.sqrt(dt_robot) * np.random.randn(t_robot.shape[0], 3) @ np.linalg.cholesky(Q_robot_discrete)

x_euler_maruyama = euler_stochatic_propagate(robot_dynamic,t_robot, robot_init, u_robot, q_robot_discrete, dt_robot)
x_robot_RK = rk4_propagate(robot_dynamic, t_robot, robot_init, u_robot, dt_robot)
x_linear = linearization_propagate(robot_dynamic,robot_jacobian, t_robot, robot_init, u_robot, q_robot, dt_robot)

x_linear = x_linear[:-1]
print('x_euler_maruyama:{},  x_robot_RK:{},  x_linear:{}'.format(np.shape(x_euler_maruyama), np.shape(x_robot_RK), np.shape(x_linear)))
skip = 120
ar = [x_euler_maruyama,x_robot_RK,x_linear]
x_final = np.mean(ar, axis=0)
print('x_final:', np.shape(x_final))
plt.plot(x_final[:, 0], x_final[:, 1], 'tab:blue', label='Averaged position', linewidth=1)
plt.quiver(x_final[::skip, 0], x_final[::skip, 1],
                    np.cos(x_final[::skip, 2]), np.sin(x_final[::skip, 2]),
                    label='Direction-averaged', linewidth=0.1, alpha=.8, color='red')
plt.quiver(robot_init[0], robot_init[1], np.cos(robot_init[-1]), np.sin(robot_init[-1]),
               linewidth=2., alpha=1.0, color='black', label='init-pose')
plt.legend()
plt.grid(True)
plt.show()
def plot():
    f, ax = plt.subplots(1,1, figsize=(12, 10))
    skip = 120

    ax.plot(x_euler_maruyama[:, 0], x_euler_maruyama[:, 1], 'tab:blue', label='Robot-position-EM', linewidth=1)
    ax.plot(x_linear[:,0],x_linear[:,1], '-r', label='Robot-position-LIN', linewidth=1)
    ax.plot(x_robot_RK[:,0],x_robot_RK[:,1], 'green', label='Robot-position-RK', linewidth=1)

    ax.quiver(x_euler_maruyama[::skip, 0], x_euler_maruyama[::skip, 1],
                    np.cos(x_euler_maruyama[::skip, 2]), np.sin(x_euler_maruyama[::skip, 2]),
                    label='Direction-EM', linewidth=0.1, alpha=.8, color='tab:blue')
    ax.quiver(x_linear[::skip,0],x_linear[::skip,1],
                   np.cos(x_linear[::skip,2]),np.sin(x_linear[::skip,2]),
                   label='Direction-LIN', linewidth=0.1, alpha=0.8, color='red')
    ax.quiver(x_robot_RK[::skip,0],x_robot_RK[::skip,1],
                   np.cos(x_robot_RK[::skip,2]),np.sin(x_robot_RK[::skip,2]),
                   label='Direction-RK', linewidth=0.1, alpha=0.8, color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.grid(True)
    ax.quiver(robot_init[0], robot_init[1], np.cos(robot_init[-1]), np.sin(robot_init[-1]),
               linewidth=2., alpha=1.0, color='black', label='init-pose')
    ax.legend()
    plt.show()



plot()
