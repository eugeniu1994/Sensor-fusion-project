import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class camera(object):
    def __init__(self, csvFile=None):
        self.F = .0 #focal length
        self.h0 = 11.5 #given cm

        self.wooden_bar = 5 #cm Distance of the wall and the wooden list/bar
        self.d = 1.7 #Distance of camera pinhole and the IR detector

        self.csvFile = csvFile

    #Task 3a
    def Calibrate_camera(self,file = 'Datasets/data/task3/camera_module_calibration_task3.csv'):
        Cam = pd.read_csv(file)
        distance_cm = np.array(Cam.iloc[:,0:1]).squeeze(-1) #from the csv file

        #get the true distance by adding wooden_bar and self.d
        distance_cm = distance_cm+self.d+self.wooden_bar
        height_px = np.array(Cam.iloc[:,1:2]).squeeze(-1)
        print('distance_cm:{}, height_px:{}'.format(np.shape(distance_cm),np.shape(height_px)))

        A_matrix = np.vstack([1 / height_px, np.ones(len(height_px))]).T

        gradient, bias = np.linalg.lstsq(A_matrix, distance_cm, rcond=None)[0]

        self.bias = bias
        self.F = gradient / self.h0
        print('Gradient ',gradient)
        print("Focal length:{}, Bias:{} ".format(self.F, self.bias))
        plt.plot(1 / height_px, distance_cm, "o", label='data')
        plt.plot(1 / height_px, gradient*(1/height_px) + bias, label='linear')
        plt.legend()
        plt.xlabel('1/height, (1/pixel)')
        plt.ylabel('dist(cm)')
        plt.show()

    def Compute_phi(self, y1, degree=False):
        phi = np.arctan2(y1, self.F)
        if degree:
            return phi #in degrees

        phi = phi*180/np.pi #converted to radias
        return phi

    def Compute_Robot_dist(self, h):
        x3 = (self.h0*self.F/h) + self.bias
        return x3

#Task 4 Motor controller
class motor(object):
    def __init__(self, csvFile=None):
        self.csvFile = csvFile

    def Compute_Robot_Speed(self,file = 'Datasets/data/task4/robot_speed_task4.csv'):
        Robot = pd.read_csv(file, header=None)
        timesteps = np.array(Robot.iloc[:, 1:2]).squeeze(-1)
        distance_cm = np.array(Robot.iloc[:, 0:1]).squeeze(-1)
        print(timesteps)
        print(distance_cm)
        t = np.linspace(0,len(timesteps), num=len(timesteps))
        print('timesteps:{}, distance_cm:{}'.format(np.shape(timesteps), np.shape(distance_cm)))
        fig, axs = plt.subplots(2)
        fig.suptitle('Position and Velocity')
        print('t {}, distance_cm:{}'.format(np.shape(t), np.shape(distance_cm)))

        timesteps_t = []
        for i in timesteps:
            cur_time = timesteps_t[-1] + i if len(timesteps_t) > 0 else i
            timesteps_t.append(cur_time)

        print(timesteps_t)

        axs[0].scatter(timesteps_t,distance_cm, label='Position')
        axs[0].set(xlabel='time', ylabel='Position')

        #velocity is the derivative of the position/distance, so
        v = 40/timesteps  #derivative of position / delta t => p(i+1)-p(i) / (t(i+1)-t(i))

        axs[1].plot(timesteps_t, v,marker='o', linestyle='dashed', label='Velocity')
        axs[1].set(xlabel='time', ylabel='Velocity')

        plt.show()

if __name__ == '__main__':
    # Task3
    csvFile = 'Datasets/data/task3/camera_reading_task3.csv'
    Camera = camera(csvFile)
    Camera.Calibrate_camera()

    #Task4
    Motor = motor()
    Motor.Compute_Robot_Speed()