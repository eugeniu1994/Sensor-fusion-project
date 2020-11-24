import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class Localization():
    def __init__(self):
        self.True_xyphi = [61,33.9,90] #x,y,\phi
        self.Init_guess = [0,0,0]
        self.h0 = 11.5
        self.f = 540.26

        self.cameraLocCSV = 'Datasets/data/task5/camera_localization_task5.csv'
        self.qr_posCSV = 'Datasets/Dataset2-20201102/qr_code_position_in_global_coordinate.csv'

        #time, qr number, Cx, Cy, w,h, di, phi
        self.cam = pd.read_csv(self.cameraLocCSV, header=None)
        #print('self.cam ', np.shape(self.cam))

        #qr_code, mid_point_x_cm, mid_point_y_cm, position_in_wall
        self.qr = pd.read_csv(self.qr_posCSV)
        #print('self.qr ', np.shape(self.qr))

        #Try to identify how many qr codes are detected by the camera at each time step
        #from the record log file.
        grouped = self.cam.groupby(self.cam.columns[0]).size()
        print(grouped)

    def localize(self):
        self.estimatedPos = []
        print('self.cam ', np.shape(self.cam))
        self.qr_numbers = list(set(self.cam.iloc[:, 1]))
        print('self.qr_numbers ',self.qr_numbers)

        for i in range(0,len(self.qr_numbers)):
            qr_1,qr_2 = self.qr_numbers[i-1], self.qr_numbers[i]
            #print('qr_1:{},qr_2:{}'.format(qr_1,qr_2))

            x_cm1, y_cm1, position_in_wall1 = self.qr.iloc[qr_1,1:]
            #print('x_cm1:{},y_cm1:{},position_in_wall1:{}'.format(x_cm1,y_cm1,position_in_wall1))
            x_cm2, y_cm2, position_in_wall2 = self.qr.iloc[qr_2, 1:]

            phi1 = np.arctan2(x_cm1, self.f)*180/np.pi
            phi2 = np.arctan2(x_cm2, self.f) * 180 / np.pi
            #print('phi1:{},phi2:{}'.format(phi1,phi2))

            #hi1 = self.h0*self.f / (np.sqrt())
            #d1 = self.h0*self.f/


        #print('self.estimatedPos ',np.shape(self.estimatedPos))
        #self.Init_guess = np.mean(self.estimatedPos , axis=0)
        #print('self.Init_guess ',self.Init_guess)

    def Estimate(self):
        x = [61,33.9,90] # the true variable
        x_0 = np.array([0, 0,0])  # set the initial guess
        iteration_end = 100  # how many iteration
        N = 1

        def g(x,s):
            di = np.sqrt((s[0]-x[0])**2 + (s[1]-x[1])**2)
            y_ = s[1]-x[1]
            x_ = s[0]-x[0]
            phi= (np.arctan2(y_,x_)) - x[2]
            return np.array([[di],[phi]])   #(2 x 1)

        def Gx(x,s): #gradient of g
            y_ = s[1] - x[1]
            x_ = s[0] - x[0]
            di = np.sqrt((x_)**2 + (y_)**2)

            Jacobian = np.array([[-x_/di, -y_/(x_**2 + y_**2)],[-y_ / di,1/(x_ + (y_**2 / x_))],[0,-1]])
            return Jacobian # (3, 2)

        def J_cost(y, x, s):
            g_ = g(x,s)
            e = (y - g_)
            return np.sum(e ** 2)

        x_history = []
        x_history.append(x_0)

        sigma = 0.001
        R_inv = np.eye(2) / (sigma * sigma)  # cache the inverse

        x_init = [0,0,0]
        y = g(x=x_init,s=[0,0]) #(2, 1)
        print('y:', np.shape(y))

        iteration_end = len(self.cam)
        # time, qr number, Cx, Cy, w,h, di, phi => convert Cx,Cy to cm
        camera_data = np.array(self.cam)
        print('camera_data ',np.shape(camera_data))

        qr_pos = np.array(self.qr)
        print('qr_pos ', np.shape(qr_pos))

        iteration_end = 10
        for j in range(iteration_end):
            print()
            qr_code = int(camera_data[j,1])
            filtered = np.array(qr_pos[(qr_pos[:,0]==qr_code)]).squeeze()
            s = [filtered[1],filtered[2]] #mid_point_x_cm,mid_point_y_cm landmark
            print('s ',s)

            gj = g(x_history[j],s)     #(2, 1) -> measurement model
            Gj = Gx(x_history[j],s).T  #(3, 2)

            #y_tilde = np.sum(y - gj, axis=0) / N  #(2, 1)
            y_tilde = y - gj    # (2, 1)

            #(Gj.T @ R_inv @ Gj) - > (3x3)
            #(Gj.T @ R_inv) @ y_tilde  - > (3x1)
            x_WLS = np.linalg.solve((Gj.T @ R_inv @ Gj), (Gj.T @ R_inv) @ y_tilde)
            x_WLS = np.array(x_WLS).squeeze()
            x_history.append(x_WLS)
            print('x_WLS ',x_WLS)


            #group by timesteps ->
            #each timestep is one measurement (more qr codes)


if __name__ == '__main__':
    loc = Localization()
    #loc.localize()
    #loc.Estimate()

    '''import sympy as sp
    from sympy import *

    x1, x2, x3, s1,s2 = sp.symbols('x1,x2,x3,s1,s2', real=True)
    f1 = sp.sqrt((s1-x1)**2 + (s2-x2)**2)             #di
    f2 = (sp.atan2((s2 - x2),(s1-x1))) - x3          #phi

    F = Matrix([f1, f2])
    print('F ', np.shape(F))
    J = F.jacobian([x1, x2, x3])
    print('J ', np.shape(J))
    print(J)
    #J = F.jacobian([x1, x2, x3]).subs([(x1, 0), (x2, 0), (x3, 0), (s1, 1), (s2, 2)])
    print()
    #print(J)
    G = np.array(J)
    print(G)'''
