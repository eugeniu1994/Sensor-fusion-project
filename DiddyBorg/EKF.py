import numpy as np
import sympy
from sympy import *
from numpy import dot, array, sqrt
from filterpy.stats import plot_covariance_ellipse
from math import sqrt, tan, cos, sin, atan2
import matplotlib.pyplot as plt

#dt = 1.0

#Jacobian of the measurement
def H_of(x, landmark_pos):
    """ compute Jacobian of H matrix where h(x) computes the range and
    bearing to a landmark for state x """

    px = landmark_pos[0]
    py = landmark_pos[1]
    hyp = (px - x[0, 0])**2 + (py - x[1, 0])**2
    dist = sqrt(hyp)

    H = array(
        [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
         [ (py - x[1, 0]) / hyp,  -(px - x[0, 0]) / hyp, -1]])
    return H

#measurement
def Hx(x, landmark_pos):
    """ takes a state variable and returns the measurement that would
    correspond to that state.
    """
    px = landmark_pos[0]
    py = landmark_pos[1]
    dist = sqrt((px - x[0, 0])**2 + (py - x[1, 0])**2)

    Hx = array([[dist],
                [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
    return Hx

from filterpy.kalman import ExtendedKalmanFilter as EKF

class RobotEKF(EKF):
    def __init__(self, dt, wheelbase, sigma_vel, sigma_steer):
        EKF.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.sigma_vel = sigma_vel
        self.sigma_steer = sigma_steer

        a, x, y, v, w, theta, time = symbols('a, x, y, v, w, theta, t')
        d = v * time
        beta = (d / w) * sympy.tan(a)
        r = w / sympy.tan(a)

        self.fxu = Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                           [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                           [theta + beta]])

        self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
        self.V_j = self.fxu.jacobian(Matrix([v, a]))

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v: 0, a: 0, time: dt, w: wheelbase, theta: 0}
        self.x_x, self.x_y, self.v, self.a, self.theta = x, y, v, a, theta

    def predict(self, u=0):
        self.x = self.move(self.x, u, self.dt)

        self.subs[self.theta] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]

        F = array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = array([[self.sigma_vel * u[0] ** 2, 0], [0, self.sigma_steer ** 2]])

        self.P = dot(F, self.P).dot(F.T) + dot(V, M).dot(V.T)

    def move(self, x, u, dt):
        h = x[2, 0]
        v = u[0]
        steering_angle = u[1]

        dist = v * dt

        if abs(steering_angle) < 0.0001:
            # approximate straight line with huge radius
            r = 1.e-30
        b = dist / self.wheelbase * tan(steering_angle)
        r = self.wheelbase / tan(steering_angle)  # radius
        sinh = sin(h)
        sinhb = sin(h + b)
        cosh = cos(h)
        coshb = cos(h + b)
        return x + array([[-r * sinh + r * sinhb],
                          [r * cosh - r * coshb],
                          [b]])

def residual(a,b):
    """ compute residual between two measurement containing [range, bearing]. Bearing
    is normalized to [0, 360)"""
    y = a - b
    if y[1] > np.pi:
        y[1] -= 2*np.pi
    if y[1] < -np.pi:
        y[1] += 2*np.pi
    return y

def run_localization(landmarks, sigma_vel=0.1, sigma_steer=np.radians(1), sigma_range=0.3, sigma_bearing=0.1, dt = 1.0, x_init = [2, 6, .3]):
    ekf = RobotEKF(dt, wheelbase=0.5, sigma_vel=sigma_vel, sigma_steer=sigma_steer)
    ekf.x = array([x_init]).T
    ekf.P = np.diag([.1, .1, .1])
    ekf.R = np.diag([sigma_range**2, sigma_bearing**2])

    sim_pos = ekf.x.copy() # simulated position
    u = array([1.1, .01]) # steering command (vel, steering angle radians)

    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)
    for i in range(200):
        sim_pos = ekf.move(sim_pos, u, dt/10.) # simulate robot
        plt.plot(sim_pos[0], sim_pos[1], ',', color='g')

        if i % 10 == 0:
            ekf.predict(u=u)

            plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=6,
                                    facecolor='b', alpha=0.08)

            x, y = sim_pos[0, 0], sim_pos[1, 0]
            for lmark in landmarks:
                d = np.sqrt((lmark[0] - x)**2 + (lmark[1] - y)**2)
                a = atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2, 0]
                z = np.array([[d + np.random.randn()*sigma_range], [a + np.random.randn()*sigma_bearing]])

                ekf.update(z, HJacobian=H_of, Hx=Hx, residual=residual,
                           args=(lmark), hx_args=(lmark))

            plot_covariance_ellipse((ekf.x[0,0], ekf.x[1,0]), ekf.P[0:2, 0:2], std=6,
                                    facecolor='g', alpha=0.4)
    plt.axis('equal')
    plt.show()
    return ekf


#-----------------------------------------------------landmarks = array([[5, 10], [10, 5], [15, 15], [20, 5]])
'''landmarks = array([[5, 10], [10, 5], [15, 15]])

ekf = run_localization(landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
                      sigma_range=0.3, sigma_bearing=0.1)
print(ekf.P.diagonal())



#-----------------------------------------------------------------
landmarks = array([[5, 10], [10, 5], [15, 15], [20, 5]])

ekf = run_localization(landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
                      sigma_range=0.3, sigma_bearing=0.1)
print(ekf.P.diagonal())

##---------------------------------------------------------------
ekf = run_localization(landmarks[0:2], sigma_vel=1.e-10, sigma_steer=1.e-10,
                       sigma_range=1.4, sigma_bearing=.05)
print(ekf.P.diagonal())
#------------------------------------------------------------

ekf = run_localization(landmarks[0:1], sigma_vel=1.e-10, sigma_steer=1.e-10,
                       sigma_range=1.4, sigma_bearing=.05)
print(ekf.P.diagonal())

#------------------------------------------------------------

landmarks = array([[5, 10], [10,  5], [15, 15], [20,  5], [15, 10],
                   [10,14], [23, 14], [25, 25], [10, 20]])

ekf = run_localization(landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
                      sigma_range=0.3, sigma_bearing=0.1)
print(ekf.P.diagonal())'''








