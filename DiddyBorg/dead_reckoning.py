import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
GPS_NOISE = np.diag([1.1, 1.1]) ** 2

show_animation = True

class getData:
    def __init__(self):
        self.index = 0
        #index = 0
        imu = 'Datasets/data/task1/imu_reading_task1.csv'
        IMU_static = pd.read_csv(imu, header=None)
        bias_omega_z = np.mean(IMU_static.iloc[:,8])
        print('bias_omega_z ',bias_omega_z)

        csvFile = 'Datasets/data/task6/imu_tracking_task6.csv'
        calib_data = np.loadtxt(csvFile, delimiter=',', skiprows=0)
        t_robot = calib_data[:, 0]
        GyZ = calib_data[:, 8]
        self.linear_xyz = calib_data[:, 1:4]
        #plt.plot(self.linear_xyz[:,0], label='acc_x')
        #plt.legend()
        #plt.show()

        w = np.array(GyZ).flatten()
        print('w ', np.shape(w))
        print('t_robot ', np.shape(t_robot))
        u_robot = []
        self.dt_robot = 0.06220197677612305
        self.dt_robot /=2
        self.vel = 10# 5.5 #maxim velocity from task1 - part 4
        self.DT = self.dt_robot # 0.1  # time tick [s]

        for i in range(1,len(t_robot)):
            dt = t_robot[i] - t_robot[i-1]

            v,w_gyro = self.vel, w[i]-bias_omega_z  #m/s, degree/s
            w_gyro = np.deg2rad(w_gyro)
            #print('dt:{}, v:{}, w:{} '.format(dt,vel,w_gyro))
            inp = [v,w_gyro] #input velocity and gyroscope
            u_robot.append(inp)
        self.landmarks, self.inputs = self.EKF_estimate2()

        self.u_robot = np.array(u_robot)
        print('u_robot ', np.shape(u_robot))

    def getPoint(self):
        v = self.u_robot[self.index, 0]
        yawrate = self.u_robot[self.index, 1]
        u = np.array([[v], [yawrate]])
        l = self.landmarks[self.index]
        #print('u ',u)
        self.index+=1
        return u, l

    def EKF_estimate2(self, imu_file=None, camera_file=None):
        import sensor_fusion as sf
        import robot_n_measurement_functions as rnmf

        camera_file = 'Datasets/data/task6/camera_tracking_task6.csv'
        imu_file = 'Datasets/data/task6/imu_tracking_task6.csv'

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
        '''[plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', s=60) for landmark in landmarks]
        phi = np.deg2rad(self.x_init[-1])
        plt.quiver(self.x_init[0], self.x_init[1], np.cos(phi), np.sin(phi),
                   linewidth=2., alpha=1.0, color='black', label='init-pose')

        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()'''

        return landmarks, inputs

csv = getData()
DT = csv.DT

def calc_input():
    #v = 1.0  # [m/s]
    #yawrate = 0.1  # [rad/s]
    #u = np.array([[v], [yawrate]])
    u = csv.getPoint()
    return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)
    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud

def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
    #print('x ', x)
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u
    return x

def observation_model_(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    #print('H ', np.shape(H)) #(2, 4)
    z = H @ x
    print('Z ', np.shape(z)) #(2, 1)
    return z

#measurement
def observation_model(x, landmark_pos):
    px = landmark_pos[0]
    py = landmark_pos[1]
    dist = np.sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    H = np.array([
            [dist],
            [math.atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]
    ])
    print('H ', np.shape(H))  # (2, 1),
    # x = (4x1)
    #z = H @ x
    return H

def jacob_f(x, u):
    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF

def jacob_h_():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ]) #(2,4)
    #print('jH ',np.shape(jH))
    return jH

#Jacobian of the measurement
def jacob_h(x,landmark_pos):
    px = landmark_pos[0]
    py = landmark_pos[1]
    hyp = (px - x[0])**2 + (py - x[1])**2
    dist = np.sqrt(hyp)

    jH = np.array([
        [-(px - x[0]) / dist, -(py - x[1]) / dist, 0, 0],
         [ (py - x[1]) / hyp,  -(px - x[0]) / hyp, -1, 0]
    ])
    print('jH ', np.shape(jH)) #(2, 4)

    return jH

def ekf_estimation(xEst, PEst, z, u, landmark=None):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()   #2x4
    print('landmark ekf_estimation :{}'.format(np.shape(landmark)))
    # compute for all of them and get the average
    landmark_pos = landmark[0]

    zPred = observation_model(xPred, landmark_pos)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

    return xEst, PEst

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")

def main():
    time = 0.0
    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    x_init = np.array([15.7, 47.5, np.deg2rad(90), 0.])
    xTrue = np.array([15.7, 47.5, np.deg2rad(90), 0.]).reshape(-1,1)
    print('xTrue ',np.shape(xTrue))
    xEst = xTrue
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning
    xDR = xTrue
    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while csv.index < len(csv.u_robot):# SIM_TIME >= time:
        #print('index ', csv.index)
        time += DT
        u,_ = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))
        show_animation = False
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            #plt.gcf().canvas.mpl_connect('key_release_event',
            #        lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            #plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    [plt.scatter(landmark[:, 0], landmark[:, 1], marker='s', s=60) for landmark in csv.landmarks]
    plt.plot(hz[0, :], hz[1, :], ".g",alpha=.2, label='measurements')
    plt.plot(hxTrue[0, :].flatten(),
                 hxTrue[1, :].flatten(), "-b")
    #plt.plot(hxDR[0, :].flatten(),
    #             hxDR[1, :].flatten(), "-k")
    plt.plot(hxEst[0, :].flatten(),
                 hxEst[1, :].flatten(), "-r", label='estimation')
        # plot_covariance_ellipse(xEst, PEst)
    phi = x_init[-2]
    plt.quiver(x_init[0], x_init[1], np.cos(phi), np.sin(phi),
               linewidth=5., alpha=1.0, color='blue', label='init-pose')

    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    main()