import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class accelerometer(object):
    def __init__(self, csvFile=None):
        self.b = []  # bias
        self.k = []  # gain
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
        #print('self.linear_xyz ', np.shape(self.linear_xyz))

        self.roll_pitch = IMU.iloc[:, 4:6]
        #print('roll_pitch ', np.shape(self.roll_pitch))

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
        #ax.set_rmax(2)
        #ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
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
            v_d = []
            v_u = []
            for v in values:
                if v > 0.5:
                    v_u.append(v)
                elif v < -0.5:
                    v_d.append(v)
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
        #print('degree_xyz', np.shape(self.degree_xyz))
        self.magnetometer_xyz = GYRO.iloc[:, 9:12]
        #print('magnetometer_xyz', np.shape(self.magnetometer_xyz))

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


    def calibrate_Gyro(self):
        degree_xyz = np.array(self.degree_xyz)
        gyro_x, gyro_y, gyro_z = degree_xyz[:, 0], degree_xyz[:, 1], degree_xyz[:, 2]
        print('gyro_x:{},gyro_y:{},gyro_z:{}'.format(np.shape(gyro_x), np.shape(gyro_y), np.shape(gyro_z)))

        min_x = min(gyro_x)
        max_x = max(gyro_x)
        min_y = min(gyro_y)
        max_y = max(gyro_y)
        min_z = min(gyro_z)
        max_z = max(gyro_z)

        print("Gyro X range: ", min_x, max_x)
        print("Gyro Y range: ", min_y, max_y)
        print("Gyro Z range: ", min_z, max_z)

        gyro_calibration = [(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2]
        print("Final calibration in deg/s:", gyro_calibration)

        fig, (uncal, cal) = plt.subplots(2, 1)
        # Clear all axis
        uncal.cla()
        t = np.linspace(0, len(gyro_x), len(gyro_x))
        # plot uncalibrated data
        uncal.plot(t, gyro_x, color='r')
        uncal.plot(t, gyro_y, color='g')
        uncal.plot(t, gyro_z, color='b')
        uncal.title.set_text("Uncalibrated Gyro")
        uncal.set(ylabel='Degrees/s')

        # plot calibrated data
        cal.plot(t, [x - gyro_calibration[0] for x in gyro_x], color='r')
        cal.plot(t, [y - gyro_calibration[1] for y in gyro_y], color='g')
        cal.plot(t, [z - gyro_calibration[2] for z in gyro_z], color='b')
        cal.title.set_text("Calibrated Gyro")
        cal.set(ylabel='Degrees/s')

        fig.tight_layout()
        fig.show()

    def calibrate_Magnetometer(self):
        magnetometer_xyz = np.array(self.magnetometer_xyz)
        mag_x, mag_y, mag_z = magnetometer_xyz[:, 0], magnetometer_xyz[:, 1], magnetometer_xyz[:, 2]
        print('mag_x:{},mag_y:{},mag_z:{}'.format(np.shape(mag_x), np.shape(mag_y), np.shape(mag_z)))

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)

        # Display the sub-plots
        ax.scatter(mag_x, mag_y, color='r')
        ax.scatter(mag_y, mag_z, color='g')
        ax.scatter(mag_z, mag_x, color='b')
        fig.show()

        min_x = min(mag_x)
        max_x = max(mag_x)
        min_y = min(mag_y)
        max_y = max(mag_y)
        min_z = min(mag_z)
        max_z = max(mag_z)

        print("X range: ", min_x, max_x)
        print("Y range: ", min_y, max_y)
        print("Z range: ", min_z, max_z)

        mag_calibration = [(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2]
        print("Final calibration in uTesla:", mag_calibration)

        cal_mag_x = [x - mag_calibration[0] for x in mag_x]
        cal_mag_y = [y - mag_calibration[1] for y in mag_y]
        cal_mag_z = [z - mag_calibration[2] for z in mag_z]

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect(1)

        # Clear all axis
        ax.cla()

        # Display the now calibrated data
        ax.scatter(cal_mag_x, cal_mag_y, color='r')
        ax.scatter(cal_mag_y, cal_mag_z, color='g')
        ax.scatter(cal_mag_z, cal_mag_x, color='b')
        fig.show()



if __name__ == '__main__':
    # csv for task 1
    csvFile = 'Datasets/data/task1/imu_reading_task1.csv'
    Accelerometer = accelerometer(csvFile)
    #Accelerometer.Visualize_Data() #Task 1a

    Gyroscope = gyroscope(csvFile)
    #Gyroscope.Vizualize_Data() #Task 1a
    Gyroscope.Compute_bias() #Task 1b
    Gyroscope.Compute_variance() #Task 1c'''

    #Task 2 --------------------------------------------------
    '''csvFile = 'Datasets/data/task2/imu_calibration_task2.csv'
    Accelerometer = accelerometer(csvFile)
    Accelerometer.Compute_Bias_Gain()

    Gyroscope = gyroscope(csvFile)
    Gyroscope.Compute_variance()
    Gyroscope.Compute_bias()'''

    #A = np.diag(Accelerometer.k)
    #b = Accelerometer.b
    #y = np.array(Accelerometer.linear_xyz)

    #Gyroscope calibration-----------------------------------------------------
    #Gyroscope.calibrate_Gyro()
    #Gyroscope.calibrate_Magnetometer()