import csv
import os
import numpy as np
from matplotlib import pyplot as plt
# import sys
# sys.path.append(".")

# task 1a
file = "/Dataset2/dataset2/data/task1/imu_reading_task1.csv"
path = os.getcwd() + file
imu = np.genfromtxt(path, delimiter=',')

t = imu[:, 0]
t = t - np.min(t)
labels = ["a_x", "a_y","a_z", "roll", "pitch", "phi_x", "phi_y", "phi_z", "m_x", "m_y", "m_z"]

# bias of linear acceleration from accelarator
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 1], label = labels[0])
ax.plot(t, imu[:, 2], label = labels[1])
ax.plot(t, imu[:, 3], label = labels[2])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of linear acceleration from accelarator')
plt.legend(loc = 1)
plt.savefig("./task1a_figure/bias_linear_acceleration_accelarator")

# bias of roll and pitch angle from accelarator
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 4], label = labels[3])
ax.plot(t, imu[:, 5], label = labels[4])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of roll and pitch angle from accelarator')
plt.legend(loc = 1)
plt.savefig("./task1a_figure/bias_roll_and_pitch_accelarator")

# bias of angular from gyroscope
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 6], label = labels[5])
ax.plot(t, imu[:, 7], label = labels[6])
ax.plot(t, imu[:, 8], label = labels[7])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of angular from gyroscope')
plt.legend(loc = 1)
plt.savefig("./task1a_figure/bias_angular_gyroscope")

# bias of magnetometer
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 9], label = labels[8])
ax.plot(t, imu[:, 10], label = labels[9])
ax.plot(t, imu[:, 11], label = labels[10])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of magnetometer')
plt.legend(loc = 1)
plt.savefig("./task1a_figure/bias_magnetometer")


# task 1b
print("The bias for gyroscope is {:.2f}, {:.2f}, {:.2f} for x, y, z respectively"
      .format(np.mean(imu[:, 6]), np.mean(imu[:, 7]), np.mean(imu[:, 8])))


# task 1c
R = np.diag(np.diag(np.cov(imu[:, 1:12].T)))
print("The variance matrix of the IMU measurement is:")
print(R)






