import csv
import os
import numpy as np
from matplotlib import pyplot as plt

# task 3a
file = "/Dataset2/dataset2/data/task3/camera_module_calibration_task3.csv"
path = os.getcwd() + file
data = np.genfromtxt(path, delimiter=',')

dist = data[:, 0] + 5 + 1.7
height = data[:, 1]

a_11 = np.sum((1./height)**2)
a_12 = np.sum(1./height)
a_21 = a_12
a_22 = height.shape[0]

b_1 = np.sum((1./height) * dist)
b_2 = np.sum(dist)

matrixA = np.array([[a_11, a_12], [a_21, a_22]])
matrixB = np.array([b_1, b_2])
matrixB = np.expand_dims(matrixB, axis=1)

[k, b] = np.linalg.inv(matrixA) @ matrixB

x = np.linspace(0.005, 0.025, 1000)
y = k * x + b

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(1./height, dist, label = "data")
ax.plot(x, y, label = "linear")
ax.set(xlabel='1/height, (1/pixel)', ylabel='dist, (cm)',
       title = ' Relation between distance of QR-codes from the camera and detected height')
plt.legend(loc = 4)
plt.savefig("./task3a_figure/Relation_distanceQR_and_detected_height")

# task 3b
h0 = 11.5
f = k / 11.5
print("The focal length is: ", f)




