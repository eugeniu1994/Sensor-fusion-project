import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class accelerometer(object):
    def __init__(self, csvFile=None):
        self.b = []  # bias
        self.gain = []  # gain
        self.linear_xyz = []  # linear acceleration x,y,z
        self.g = 9.8
        self.roll_pitch = []  # roll and pitch angles
        self.timesteps = []
        self.csvFile = csvFile

        self.read_file()

    def read_file(self):
        IMU = pd.read_csv(self.csvFile)
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

    #Task2
    def Compute_Bias_Gain(self):
        x = np.array(self.linear_xyz.iloc[:, 0])
        y = np.array(self.linear_xyz.iloc[:, 1])
        z = np.array(self.linear_xyz.iloc[:, 2])

        def find_value_of_interest(values): #found from other implementation
            v_d = []
            v_u = []
            for v in values:
                if v > 0.5:
                    v_u.append(v)
                elif v < -0.5:
                    v_d.append(v)
            return np.mean(v_u), np.mean(v_d)

        x_u, x_d = find_value_of_interest(x)  # values up and down for x
        y_u, y_d = find_value_of_interest(y)  # values up and down for x
        z_u, z_d = find_value_of_interest(z)  # values up and down for x

        # according to given formula
        self.b = [(x_u + x_d) / 2, (y_u + y_d) / 2, (z_u + z_d) / 2]
        self.gain = [(x_u - x_d) / (2 * self.g), (y_u - y_d) / (2 * self.g), (z_u - z_d) / (2 * self.g)]

        print(self.b)
        print(self.gain)

class gyroscope(object):
    def __init__(self, csvFile=None):
        self.timesteps = []
        self.degree_xyz = []
        self.magnetometer_xyz = []
        self.csvFile = csvFile

        self.read_file()

    def read_file(self):
        GYRO = pd.read_csv(self.csvFile)
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
        plt.legend(["X", "Y", "Z"])
        plt.xlabel("Timestamp")
        plt.ylabel("Angular velocity [degree/s]")
        plt.show()

    def Compute_bias(self):
        x = np.array(self.degree_xyz.iloc[:, 0]).mean()
        y = np.array(self.degree_xyz.iloc[:, 1]).mean()
        z = np.array(self.degree_xyz.iloc[:, 2]).mean()

        b = [x, y,z]
        print('b ', b)
        return b

    def Compute_variance(self):
        cov = np.cov(self.degree_xyz, rowvar=False)
        print('cov ',cov)
        return cov

if __name__ == '__main__':
    # csv for task 1
    csvFile = 'Datasets/data/task1/imu_reading_task1.csv'
    #Accelerometer = accelerometer(csvFile)
    #Accelerometer.Visualize_Data() #Task 1a

    #Gyroscope = gyroscope(csvFile)
    #Gyroscope.Vizualize_Data() #Task 1a
    #Gyroscope.Compute_bias() #Task 1b
    #Gyroscope.Compute_variance() #Task 1c

    #Task 2 --------------------------------------------------
    #csvFile = 'Datasets/data/task2/imu_calibration_task2.csv'
    #Accelerometer = accelerometer(csvFile)
    #Accelerometer.Compute_Bias_Gain()

    # import modules
    from scipy.integrate import cumtrapz
    from numpy import sin, cos, pi
    #from mpl_toolkits import mplot3d

    csvFile = 'Datasets/data/task2/imu_calibration_task2.csv'
    print('csvFile ',csvFile)
    plt.style.use('seaborn')  # import data from CSV
    df = pd.read_csv(csvFile)
    # Take a look at all sensor outputs
    df.plot(subplots=True, sharex=True, layout=(4, 3))
    #df.plot()

    plt.show()

    #---------------------------------------------------
    # Convert orientation measurement units to radians
    cols_angles = [4,5]
    for axis in cols_angles:
        print('axis ',axis)
        df.iloc[:,axis] = df.iloc[:,axis] * pi / 180  # Transform body frame accelerations into the inertial (Earth) frame

    def R_x(x):
        return np.array([[1, 0, 0],
                         [0, cos(-x), -sin(-x)],
                         [0, sin(-x), cos(-x)]])

    def R_y(y):
        return np.array([[cos(-y), 0, -sin(-y)],
                        [0, 1, 0],
                         [sin(-y), 0, cos(-y)]])

    def R_z(z):
        return np.array([[cos(-z), -sin(-z), 0],
                     [sin(-z), cos(-z), 0],
                     [0, 0, 1]])

    # Set up arrays to hold acceleration data for transformation
    accel = np.array([df.iloc[:,1],
                      df.iloc[:,2],
                      df.iloc[:,3]])

    pitch = df.iloc[:,5]
    roll = df.iloc[:,4]
    #yaw = df['ORIENTATION Z (azimuth °)']  # Initilize arrays for new transformed accelerations
    earth_linear = np.empty(accel.shape)  # Perform frame transformations (body frame --> earth frame)

    for i in range(df.shape[0]):
        # accel_earth = (RzRyRx)(accel_body)
        earth_linear[:, i] = R_z(0) @ R_y(roll[i]) @ R_x(pitch[i]) @ accel[:, i]

    df['EARTH LINEAR ACCELERATION X'] = earth_linear[0, :]
    df['EARTH LINEAR ACCELERATION Y'] = earth_linear[1, :]
    df['EARTH LINEAR ACCELERATION Z'] = earth_linear[2, :]


    dt = .01# df.iloc[:,0]
    dt = 0.01  # Sampling at 100Hz
    # Double integrate accelerations to find positions
    x = cumtrapz(cumtrapz(df['EARTH LINEAR ACCELERATION X'], dx=dt), dx=dt)
    y = cumtrapz(cumtrapz(df['EARTH LINEAR ACCELERATION Y'], dx=dt), dx=dt)
    z = cumtrapz(cumtrapz(df['EARTH LINEAR ACCELERATION Z'], dx=dt), dx=dt)  # Plot 3D Trajectory

    fig3, ax = plt.subplots()
    fig3.suptitle('3D Trajectory of phone', fontsize=20)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, c='red', lw=5, label='phone trajectory')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')

    plt.show()

    #-------------------------------------------------------------------------------
    # Try to remove noise via Fourier analysis
    # Discrete Fourier Transform sample frequencies
    freq = np.fft.rfftfreq(df['EARTH LINEAR ACCELERATION X'].size, d=dt)
    # Compute the Fast Fourier Transform (FFT) of acceleration signals
    fft_x = np.fft.rfft(df['EARTH LINEAR ACCELERATION X'])
    fft_y = np.fft.rfft(df['EARTH LINEAR ACCELERATION Y'])
    fft_z = np.fft.rfft(df['EARTH LINEAR ACCELERATION Z'])  # Plot Frequency spectrum
    fig4, [ax1, ax2, ax3] = plt.subplots(3, 1, sharex=True, sharey=True)
    fig4.suptitle('Noise Spectrum', fontsize=20)
    ax1.plot(freq, abs(fft_x), c='r', label='x noise')
    ax1.legend()
    ax2.plot(freq, abs(fft_y), c='b', label='y noise')
    ax2.legend()
    ax3.plot(freq, abs(fft_z), c='g', label='z noise')
    ax3.legend()
    ax3.set_xlabel('Freqeuncy (Hz)')
    plt.show()


    #---------------------------------------------------------------------

    # Add XYZ axis arrows to indicate phone pose
    # Earth 3 axis vectors
    earth_x = np.array([[1, 0, 0], ] * len(x)).T
    earth_y = np.array([[0, 1, 0], ] * len(x)).T
    earth_z = np.array([[0, 0, 1], ] * len(x)).T

    # Initilize body Vectors
    body_x = np.empty(earth_x.shape)
    body_y = np.empty(earth_y.shape)
    body_z = np.empty(earth_z.shape)

    # Perform inverse frame transformations (body frame <-- earth frame)
    # body_vectors = (RxRyRz)(earth_vectors)
    for i in range(x.shape[0]):
        # use negative angles to reverse rotation
        body_x[:, i] = R_x(-pitch[i]) @ R_y(-roll[i]) @ R_z(-0) @ earth_x[:, i]
        body_y[:, i] = R_x(-pitch[i]) @ R_y(-roll[i]) @ R_z(-0) @ earth_y[:, i]
        body_z[:, i] = R_x(-pitch[i]) @ R_y(-roll[i]) @ R_z(-0) @ earth_z[:, i]

    # Set length of quiver arrows
    distance = np.sqrt(x[-1] ** 2 + y[-1] ** 2 + z[-1] ** 2)
    length = 1.5# 0.05 * distance
    # Plot x vectors
    # downsampling to every 10th arrow ([::10])
    fig6, ax4 = plt.subplots()
    fig6.suptitle('Phone trajectory and pose', fontsize=20)
    ax4 = plt.axes(projection='3d')
    ax4.plot3D(x, y, z, 'k', lw=5, label='Attenuated phone trajectory')
    size = 20
    ax4.quiver(x[::size], y[::size], z[::size],
               body_x[0][::size], body_x[1][::size], body_x[2][::size],
               color='b', label='x axis', length=length)
    # Plot y vectors
    ax4.quiver(x[::size], y[::size], z[::size],
               body_y[0][::size], body_y[1][::size], body_y[2][::size],
               color='r', label='y axis', length=length)
    # Plot Z vectors
    ax4.quiver(x[::size], y[::size], z[::size],
               body_z[0][::size], body_z[1][::size], body_z[2][::size],
               color='g', label='z axis', length=length)
    ax4.set_xlabel('X position (m)')
    ax4.set_ylabel('Y position (m)')
    ax4.set_zlabel('Z position (m)')
    #ax4.set_xlim(-1, 1)
    #ax4.set_ylim(-1, 1)
    #ax4.set_zlim(-1.3, 0.7)
    ax4.legend(fontsize='x-large')

    plt.show()


