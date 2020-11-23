import csv
import os
import numpy as np
from matplotlib import pyplot as plt

# task 3a
file = "/Dataset2/dataset2/data/task4/robot_speed_task4.csv"
path = os.getcwd() + file
data = np.genfromtxt(path, delimiter=',')

speed = 40.0 / data[:, 1]
print(speed)





