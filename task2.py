import csv
import os
import numpy as np
from matplotlib import pyplot as plt
# import sys
# sys.path.append(".")

# task 2a
file = "/Dataset2/dataset2/data/task2/imu_calibration_task2.csv"
path = os.getcwd() + file
imu = np.genfromtxt(path, delimiter=',')

t = imu[:, 0]
t = t - np.min(t)
labels = ["a_x", "a_y","a_z", "roll", "pitch", "phi_x", "phi_y", "phi_z", "m_x", "m_y", "m_z"]

timesteps = np.arange(t.shape[0])

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 1], label = labels[0])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of linear x acceleration from accelarator')
plt.legend(loc = 1)
plt.savefig("./task2a_figure/bias_linear_x_acceleration_accelarator")

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 2], label = labels[1])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of linear y acceleration from accelarator')
plt.legend(loc = 1)
plt.savefig("./task2a_figure/bias_linear_y_acceleration_accelarator")

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 3], label = labels[2])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of linear z acceleration from accelarator')
plt.legend(loc = 1)
plt.savefig("./task2a_figure/bias_linear_z_acceleration_accelarator")

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t, imu[:, 1], label = labels[0])
ax.plot(t, imu[:, 2], label = labels[1])
ax.plot(t, imu[:, 3], label = labels[2])
ax.set(xlabel='time (s)', ylabel='bias', title='bias of linear acceleration from accelarator')
plt.legend(loc = 1)
plt.savefig("./task2a_figure/bias_linear_acceleration_accelarator")
# plt.show()

accXup = imu[786:1204, 1]
accXdown = imu[1324:1700, 1]
accYup = imu[1809:2167, 2]
accYdown = imu[2314:2744, 2]
accZup = imu[10:257, 3]
accZdown = imu[373:723, 3]

kX = 0.5 * (np.mean(accXup) - np.mean(accXdown))
kY = 0.5 * (np.mean(accYup) - np.mean(accYdown))
kZ = 0.5 * (np.mean(accZup) - np.mean(accZdown))

bX = 0.5 * (np.mean(accXup) + np.mean(accXdown))
bY = 0.5 * (np.mean(accYup) + np.mean(accYdown))
bZ = 0.5 * (np.mean(accZup) + np.mean(accZdown))

print("k_x: ", kX)
print("k_y: ", kY)
print("k_z: ", kZ)

print("b_x: ", bX)
print("b_y: ", bY)
print("b_z: ", bZ)

# task 2b
x_positive_estimated = np.sum(accXup - bX) / (accXup.shape[0] * kX)
x_negative_estimated = np.sum(accXdown - bX) / (accXdown.shape[0] * kX)

y_positive_estimated = np.sum(accYup - bY) / (accYup.shape[0] * kY)
y_negative_estimated = np.sum(accYdown - bY) / (accYdown.shape[0] * kY)

z_positive_estimated = np.sum(accZup - bZ) / (accZup.shape[0] * kZ)
z_negative_estimated = np.sum(accZdown - bZ) / (accZdown.shape[0] * kZ)

print("+x: ", x_positive_estimated)
print("-x: ", x_negative_estimated)
print("+y: ", y_positive_estimated)
print("-y: ", y_negative_estimated)
print("+z: ", z_positive_estimated)
print("-z: ", z_negative_estimated)