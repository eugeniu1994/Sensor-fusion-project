import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class accelerometer(object):
    def __init__(self, csvFile=None):
        self.b = []                     # bias
        self.k = []                     # gain
        self.linear_xyz = []  # linear acceleration x,y,z
        self.g = 9.8
        self.roll_pitch = []  # roll and pitch angles
        self.timesteps = []
        self.csvFile = csvFile

        self.read_file()

    def read_file(self):
        IMU = pd.read_csv(self.csvFile, header=None)
        self.timesteps = IMU.iloc[:, 0:1]
        self.linear_xyz = IMU.iloc[:, 1:4]
        self.roll_pitch = IMU.iloc[:, 4:6]

    # Task 1.a
    def Visualize_Data(self):
        # plot the linear acceleration x,y,z
        plt.plot(self.linear_xyz.iloc[:, 0])
        plt.plot(self.linear_xyz.iloc[:, 1])
        plt.plot(self.linear_xyz.iloc[:, 2])
        plt.ylabel("linear acceleration in gravity unit")
        plt.xlabel("Timestamp ")
        plt.legend(["X", "Y", "Z"])
        plt.show()

        #plot roll and pitch
        r = np.linspace(0, 2, num=len(self.roll_pitch.iloc[:,0]), endpoint=False)
        roll = np.array(self.roll_pitch.iloc[:,0])
        roll_rad = [ math.radians(degree) for degree in roll]
        pitch = np.array(self.roll_pitch.iloc[:, 1])
        pitch_rad = [math.radians(degree) for degree in pitch]

        ax = plt.subplot(111, projection='polar')
        ax.plot(roll_rad, r, label='roll')
        ax.plot(pitch_rad, r, label='pitch')
        ax.set_rmax(2)
        ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.legend()
        ax.set_title("Roll and Pitch", va='bottom')
        plt.show()

        ax = plt.subplot(111)
        ax.plot(self.roll_pitch.iloc[:,0], label='roll')
        ax.plot(self.roll_pitch.iloc[:,1], label='pitch')
        ax.set_ylim([-180, 180])
        ax.set_ylabel("Roll and Pitch in degrees")
        ax.set_xlabel("Timestamp ")
        ax.grid(True)
        ax.legend()
        ax.set_title("Roll and Pitch", va='bottom')
        plt.show()

    #Task2
    def Compute_Bias_Gain(self):
        x = np.array(self.linear_xyz.iloc[:, 0])
        y = np.array(self.linear_xyz.iloc[:, 1])
        z = np.array(self.linear_xyz.iloc[:, 2])

        def get_Data(values): #found from other implementation
            v_d,v_u = [],[]
            for i in values:
                if i > 0.5:
                    v_u.append(i)
                elif i < -0.5:
                    v_d.append(i)
            return np.mean(v_u), np.mean(v_d)

        x_u, x_d = get_Data(x)  #  up and down for x
        y_u, y_d = get_Data(y)  #  up and down for y
        z_u, z_d = get_Data(z)  #  up and down for z

        # according to given formula
        self.b = [(x_u + x_d) / 2, (y_u + y_d) / 2, (z_u + z_d) / 2]
        self.k = [(x_u - x_d) / (2 * self.g), (y_u - y_d) / (2 * self.g), (z_u - z_d) / (2 * self.g)]

class gyroscope(object):
    def __init__(self, csvFile=None):
        self.timesteps = []
        self.degree_xyz = []
        self.magnetometer_xyz = []
        self.csvFile = csvFile

        self.R = None #variance
        self.bias = None

        self.read_file()

    def read_file(self):
        GYRO = pd.read_csv(self.csvFile, header=None)
        self.timesteps = GYRO.iloc[:, 0:1]
        self.degree_xyz = GYRO.iloc[:, 6:9]
        self.magnetometer_xyz = GYRO.iloc[:, 9:12]

    # Task 1.a
    def Vizualize_Data(self):
        x = np.array(self.degree_xyz.iloc[:, 0])
        y = np.array(self.degree_xyz.iloc[:, 1])
        z = np.array(self.degree_xyz.iloc[:, 2])

        # plot the angular velocity x,y,z
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        plt.ylim([-180, 180])
        plt.legend(["X", "Y", "Z"])
        plt.xlabel("Timestamp")
        plt.ylabel("Angular velocity [degree/s]")
        plt.show()

        x = np.array(self.magnetometer_xyz.iloc[:, 0])
        y = np.array(self.magnetometer_xyz.iloc[:, 1])
        z = np.array(self.magnetometer_xyz.iloc[:, 2])
        # plot the magnetometer_xyz
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        plt.legend(["X", "Y", "Z"])
        plt.xlabel("Timestamp")
        plt.ylabel("magnetometer")
        plt.show()

    def Compute_bias(self):
        x = np.array(self.degree_xyz.iloc[:, 0]).mean()
        y = np.array(self.degree_xyz.iloc[:, 1]).mean()
        z = np.array(self.degree_xyz.iloc[:, 2]).mean()

        b = [x, y,z]
        self.bias = b
        print('bias: ',self.bias)

    def Compute_variance(self, diag = True):
        R = np.cov(self.degree_xyz, rowvar=False)
        if diag:
            self.R = R
            print(self.R)
            return np.diag(np.diag(R))
        else:
            self.R = R
            print(self.R)
            return R

if __name__ == '__main__':
    # csv for task 1 ----------------------------------------
    csvFile = 'Datasets/data/task1/imu_reading_task1.csv'
    Accelerometer = accelerometer(csvFile)
    Accelerometer.Visualize_Data()

    Gyroscope = gyroscope(csvFile)
    Gyroscope.Vizualize_Data()
    Gyroscope.Compute_bias()
    Gyroscope.Compute_variance()

    #Task 2 --------------------------------------------------
    csvFile = 'Datasets/data/task2/imu_calibration_task2.csv'
    Accelerometer = accelerometer(csvFile)
    Accelerometer.Compute_Bias_Gain()
    Accelerometer.Visualize_Data()
    print('b ', Accelerometer.b)
    print('k ', Accelerometer.k)
    b = Accelerometer.b

    A = np.diag(Accelerometer.k)
    y = np.array(Accelerometer.linear_xyz)
    b = np.array(b)[:, np.newaxis]
    G = np.linalg.inv(A)
    x = np.array(Accelerometer.linear_xyz).T
    y = G @ x + b
    y_tilde = y - b

    a1 = G.T @ G
    a2 = G.T @ y_tilde
    x_WLS = np.linalg.solve(a1, a2)
    print(np.sum(x_WLS-x)**2)