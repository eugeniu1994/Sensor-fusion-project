
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.linalg as sla
#sns.set('talk')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

fig_size = (8,8)
from typing import Callable
np.random.seed(2020)
skip = 150

def Extended_Kalman_filter(f:Callable[[np.ndarray,np.ndarray],np.ndarray],h:Callable[[np.ndarray],np.ndarray],
                  df_dx:Callable[[np.ndarray,np.ndarray],np.ndarray],dh_dx:Callable[[np.ndarray],np.ndarray],
                  x_init,cov_init,Q_discrete,R_discrete,y,u=None):
    length = y.shape[0]
    x = np.zeros((length,x_init.shape[0]))
    P = np.zeros((length,x_init.shape[0],x_init.shape[0]))
    x[0] = x_init
    P[0] = cov_init
    
    if u is None:
        u = np.zeros((length,1))
        
    for i in range(length-1):
        x_     = f(x[i],u[i])
        F_x    = df_dx(x[i],u[i])
        P_     = F_x@P[i]@F_x.T + Q_discrete
        
        H_x    = dh_dx(x_)
        S      = H_x@P_@H_x.T + R_discrete
        K_tran = sla.solve(S,H_x@P_)
        P[i+1] = P_ - K_tran.T@S@K_tran
        x[i+1] = x_ + K_tran.T@(y[i]-h(x_))
    return x, P



# ### The unscented Kalman filter
def prepare_unscented_weigths(n_state:int, alpha:float, beta:float):
    kappa = 3 - n_state
    lamb = -n_state + (n_state+kappa)*alpha**2 
    wm_0 = lamb / (n_state+ lamb)
    wp_0 = lamb / (n_state+ lamb) + (1- alpha**2 + beta)
    wm_else = 1/(2*(n_state+lamb))
    wp_else = wm_else
    WM = np.zeros(2*n_state+1)
    WP = np.zeros(2*n_state+1)
    for j in range(2*n_state + 1):
        if j==0:
            WM[j] = wm_0
            WP[j] = wp_0
        else:
            WM[j] = wm_else
            WP[j] = wp_else
    
    return lamb,WM,WP

def unscented_transformation(x:np.ndarray,P:np.ndarray,lamb:float):
    sqrt_P    = np.linalg.cholesky(P)
    return x + np.sqrt(x.shape[0]+lamb)*np.vstack([np.zeros((1,x.shape[0])),sqrt_P,-sqrt_P])

def _outer(x:np.ndarray,y:np.ndarray):
    return np.outer(x,y)
outer_prod = np.vectorize(_outer,signature="(n),(m)->(n,m)")

def compute_covariance(xs:np.ndarray,ys:np.ndarray,weight:np.ndarray):
    unnormalized = outer_prod(xs,ys)
    return np.tensordot(weight,unnormalized,axes=([0],[0]))

def unscented_Kalman_filter(f:Callable[[np.ndarray],np.ndarray],h:Callable[[np.ndarray],np.ndarray],
                  WM:np.ndarray, WP:np.ndarray,lamb:float,
                  x_init,cov_init,Q_discrete,R_discrete,y,u=None):
    length = y.shape[0]
    x = np.zeros((length,x_init.shape[0]))
    P = np.zeros((length,x_init.shape[0],x_init.shape[0]))
    x[0] = x_init
    P[0] = cov_init
    if u is None:
        u = np.zeros((length,1))
    for i in range(length-1):
        x_ust = unscented_transformation(x[i],P[i],lamb)
        f_ust = f(x_ust,u[i])
        x_hat_min = WM@f_ust
        P_min = compute_covariance(f_ust-x_hat_min,f_ust-x_hat_min,WP)+Q_discrete
        
        #this is additional step mentioned in the lecture slide
        #The x_ust only used to compute h_ust
        x_ust = unscented_transformation(x_hat_min,P_min,lamb)
        x_ = WM@x_ust
        #-----------
        
        
        h_ust = h(x_ust)
        h_ = WM@h_ust
        S_ = compute_covariance(h_ust-h_,h_ust-h_,WP)+R_discrete
        cov_h_f = compute_covariance(h_ust-h_,x_ust-x_hat_min,WP)
        K_tran = sla.solve(S_,cov_h_f)
        
        #The remaining is just the ordinary kalman filter
        P[i+1] = P_min - K_tran.T@S_@K_tran
        x[i+1] = x_hat_min + K_tran.T@(y[i]-h_)        
    return x, P

# ### Question 2 (Robot model with direct position measurements)
# Recall the Euler–Maruyama discretization of the 2D dynamic model of a robot platform in Exercise 8.3.
# 
# a. Assume that we measure the position of the robot with additive Gaussian noise. Form a state-space model for the system.
# 
# b. Simulate states and measurements from the model and plot them.
# 
# c. Derive and implement EKF for the model. Plot the result.

'''robot_init = np.array([0.5,0.5,np.pi/4])
t_robot = np.linspace(0.,5.,501)
dt_robot = t_robot[1]-t_robot[0]
u_robot = np.zeros((t_robot.shape[0],2))

for i in range(t_robot.shape[0]):
    if 0<= t_robot[i] < 1:
        u_robot[i,0] = t_robot[i]
    elif 1<= t_robot[i] < 4:
        u_robot[i,0] = 1
    else:
        u_robot[i,0] = 5-t_robot[i]
        
    if 0<= t_robot[i] < 2:
        u_robot[i,1] = 0
    elif 2<= t_robot[i] < 3:
        u_robot[i,1] = np.pi
    else:
        u_robot[i,1] = 0'''
robot_init = np.array([15.7, 47.5, np.deg2rad(90)])
imu = 'Datasets/data/task1/imu_reading_task1.csv'
IMU_static = pd.read_csv(imu, header=None)
bias_omega_z = np.mean(IMU_static.iloc[:,8])
print('bias_omega_z ',bias_omega_z)

csvFile = 'Datasets/data/task6/imu_tracking_task6.csv'
calib_data = np.loadtxt(csvFile, delimiter=',', skiprows=0)
t_robot = calib_data[:, 0]
GyZ = calib_data[:, 8]

w = np.array(GyZ).flatten()
print('w ', np.shape(w))
print('t_robot ', np.shape(t_robot))
u_robot = []
dt_robot = 0.06220197677612305
dt_robot /=2
vel = 5.5 #maxim velocity from task1 - part 4

for i in range(1,len(t_robot)):
    dt = t_robot[i] - t_robot[i-1]
    v,w_gyro = vel, w[i]-bias_omega_z  #m/s, degree/s
    w_gyro = np.deg2rad(w_gyro)
    #print('dt:{}, v:{}, w:{} '.format(dt,vel,w_gyro))
    inp = [v,w_gyro] #input velocity and gyroscope
    u_robot.append(inp)

u_robot = np.array(u_robot)
print('u_robot ', np.shape(u_robot))


Q_robot = np.diag([1e-2,1e-2,1e-2])
Q_robot_discrete = dt_robot*Q_robot
#q_robot_discrete = np.random.multivariate_normal(np.zeros(3),Q_robot_discrete,(t_robot.shape[0]))

q_robot = np.random.randn(t_robot.shape[0], 3) @ np.linalg.cholesky(Q_robot)
q_robot_discrete = np.sqrt(dt_robot) * np.random.randn(t_robot.shape[0], 3) @ np.linalg.cholesky(Q_robot_discrete)


R_robot_discrete = 0.01*np.eye(2)
r_robot = np.random.multivariate_normal(np.zeros(2),R_robot_discrete,(t_robot.shape[0]))


def _f_robot(x:np.ndarray,u:np.ndarray):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])

f_robot = np.vectorize(_f_robot,signature='(n),(m)->(n)')

def f_robot_discrete(x:np.ndarray,u:np.ndarray):
    return x + f_robot(x,u)*dt_robot

def df_dx_robot(x:np.ndarray,u:np.ndarray):
    jac = np.zeros((x.shape[0],x.shape[0]))
    jac[0,2] = -u[0]*np.sin(x[2])
    jac[1,2] = u[0]*np.cos(x[2])
    
    return jac

def df_dx_robot_discrete(x:np.ndarray,u:np.ndarray):
    return np.eye(x.shape[0])+df_dx_robot(x,u)*dt_robot

def _h_robot(x:np.ndarray):
    return np.array([x[0],x[1]])

h_robot = np.vectorize(_h_robot, signature='(n)->(2)') ##Vectorize the stuff to avoid doing looping

def dh_dx_robot(x:np.ndarray):
    return np.array([[1,0.,0.],[0.,1.,0.]])

def euler_maruyama_propagate(f:Callable[[np.ndarray,np.ndarray],np.ndarray], 
                              x_init:np.ndarray, w:np.ndarray, dt:float, u:np.ndarray=None):
    length = w.shape[0]
    if u is None:
        u = np.zeros((length,1))
    x_res = np.zeros((length,x_init.shape[0]))
    x_res[0] = x_init
    for i in range(x_res.shape[0]-1):
        x_res[i+1] = x_res[i] + f(x_res[i],u[i])*dt +  w[i]
    return x_res

x_robot = euler_maruyama_propagate(f_robot, robot_init, q_robot_discrete, dt_robot, u_robot)


# ### The measurement

y_robot = h_robot(x_robot)+r_robot
plt.figure()
plt.plot(y_robot[:,0],y_robot[:,1], 'ok', markersize=6, linewidth=0.5,alpha=0.25)
plt.show()

x_robot_ekf_estimate,P_robot_ekf_estimate = Extended_Kalman_filter(f_robot_discrete,h_robot,df_dx_robot_discrete,dh_dx_robot,
                                               robot_init,Q_robot_discrete,Q_robot_discrete,
                                               R_robot_discrete,y_robot,u_robot)
stdev_robot_ekf = np.sqrt(np.diagonal(P_robot_ekf_estimate,axis1=1,axis2=2))

lamb_robot,WM_robot,WP_robot = prepare_unscented_weigths(n_state=robot_init.shape[0], alpha=1, beta=0)
x_robot_ukf_estimate,P_robot_ukf_estimate = unscented_Kalman_filter(f_robot_discrete,h_robot,WM_robot,WP_robot,lamb_robot,
                                               robot_init,Q_robot_discrete,Q_robot_discrete,
                                               R_robot_discrete,y_robot,u_robot)
stdev_robot_ukf = np.sqrt(np.diagonal(P_robot_ukf_estimate,axis1=1,axis2=2))

f, ax = plt.subplots(1,3,figsize=(2*fig_size[0],fig_size[1]))
for i in range(3):
    ax[i].plot(t_robot,x_robot_ukf_estimate[:,i],linewidth=0.5, label='$\hat x_{}$ UKF'.format(i+1))
    ax[i].plot(t_robot,x_robot_ekf_estimate[:,i],linewidth=0.5, color='red', label='$\hat x_{}$ EKF'.format(i+1))
    ax[i].fill_between(t_robot,x_robot_ukf_estimate[:,i]-2*stdev_robot_ukf[:,i],x_robot_ukf_estimate[:,i]+2*stdev_robot_ukf[:,i],label='conf. UKF \n $x_{}$'.format(i+1),color='b',alpha=0.2)
    ax[i].fill_between(t_robot,x_robot_ekf_estimate[:,i]-2*stdev_robot_ekf[:,i],x_robot_ekf_estimate[:,i]+2*stdev_robot_ekf[:,i],label='conf. EKF \n $x_{}$'.format(i+1),color='r',alpha=0.2)
    ax[i].plot(t_robot,x_robot[:,i],'-k',linewidth=0.5, label='$x_{}$'.format(i+1))
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$x$')
    ax[i].legend()
plt.show()

f, ax = plt.subplots(2,2, figsize=(2*fig_size[0],2*fig_size[1]))
quiver_width=0.005
quiver_head_width=3.
quiver_head_length=4.

ax[0,0].plot(u_robot[:,0], label='Velocity', linewidth=0.5)
ax[0,0].set_xlabel('$t$')
ax[0,0].set_ylabel('$v$')
ax[0,0].legend()

ax[0,1].plot(u_robot[:,1], label='Gyroscope', linewidth=0.5)
ax[0,1].set_xlabel('$t$')
ax[0,1].set_ylabel('$\omega / \pi$')
ax[0,1].legend()

ax[1,0].plot(x_robot[:,0],x_robot[:,1], '-k', label='Robot-position-EM', linewidth=0.5)
ax[1,0].quiver(x_robot[::skip,0],x_robot[::skip,1],
               np.cos(x_robot[::skip,2]),np.sin(x_robot[::skip,2]),
               label='Direction-EM', linewidth=0.5, alpha=0.5, color='black',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(x_robot_ekf_estimate[:,0],x_robot_ekf_estimate[:,1], '-r', label='Robot-position-EKF', linewidth=0.5)
ax[1,0].quiver(x_robot_ekf_estimate[::skip,0],x_robot_ekf_estimate[::skip,1],
               np.cos(x_robot_ekf_estimate[::skip,2]),np.sin(x_robot_ekf_estimate[::skip,2]),
               label='Direction-EKF', linewidth=0.5, alpha=0.5, color='red',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(x_robot_ukf_estimate[:,0],x_robot_ukf_estimate[:,1], '-b', label='Robot-position-UKF', linewidth=0.5)
ax[1,0].quiver(x_robot_ukf_estimate[::skip,0],x_robot_ukf_estimate[::skip,1],
               np.cos(x_robot_ukf_estimate[::skip,2]),np.sin(x_robot_ukf_estimate[::skip,2]),
               label='Direction-UKF', linewidth=0.5, alpha=0.5, color='blue',
               width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(y_robot[:,0],y_robot[:,1], 'ok', markersize=1, label='Measurement', alpha=0.2)
ax[1,0].set_xlabel('$x_1$')
ax[1,0].set_ylabel('$x_2$')
ax[1,0].legend()

ax[1,1].plot(t_robot,x_robot[:,2]/np.pi, '-k', label='Orientation-EM', linewidth=0.5)
ax[1,1].plot(t_robot,x_robot_ekf_estimate[:,2]/np.pi, '-r', label='Orientation-EKF', linewidth=0.5)
ax[1,1].plot(t_robot,x_robot_ukf_estimate[:,2]/np.pi, '-b', label='Orientation-UKF', linewidth=0.5)
ax[1,1].set_xlabel('$t$')
ax[1,1].set_ylabel('$\phi / \pi$')
ax[1,1].legend()

plt.show()

# ## Question 3 (Robot model with distance and bearing measurements)
# Assume that in the robot model in the previous exercise, instead of position, we measure the distance and bearing (= direction) to a landmark in position $(s_x , s_y )$. Simulate data from the model, derive and implement EKF for it, and
# visualize the results. Please note that you need to take special care that the bearing measurement prediction and measurement are in the same quadrant.
# ## Assume that we have $(s_x , s_y ) = (0,0)$
s = [0,0]
def _h_robot_2(x:np.ndarray):
    g = np.array([np.sqrt(((x[0]-s[0])**2) + ((x[1]-s[1])**2)),np.arctan2(s[1]-x[1],s[0]-x[0])-x[2]])
    #return g
    return np.array([np.sqrt(x[0]*x[0]+x[1]*x[1]),np.arctan2(-x[1],x[0])-x[2]])

h_robot_2 = np.vectorize(_h_robot_2, signature='(n)->(2)') ##Vectorize the stuff to avoid doing looping
h_robot_2 = np.vectorize(_h_robot, signature='(n)->(2)') ##Vectorize the stuff to avoid doing looping

def dh_dx_robot_2(x:np.ndarray):
    distance = np.sqrt(x[0]*x[0]+x[1]*x[1])
    return np.array([[x[0]/distance,x[1]/distance, 0.],
                     [-x[1]/distance,x[0]/distance, -1]])

x_robot_ekf_estimate_2,P_robot_ekf_estimate_2 = Extended_Kalman_filter(f_robot_discrete,h_robot_2,df_dx_robot_discrete,dh_dx_robot_2,
                                               robot_init,Q_robot_discrete,Q_robot_discrete,
                                               R_robot_discrete,y_robot,u_robot)
stdev_robot_ekf_2 = np.sqrt(np.diagonal(P_robot_ekf_estimate_2,axis1=1,axis2=2))


x_robot_ukf_estimate_2,P_robot_ukf_estimate_2 = unscented_Kalman_filter(f_robot_discrete,h_robot_2,WM_robot,WP_robot,lamb_robot,
                                               robot_init,Q_robot_discrete,Q_robot_discrete,
                                               R_robot_discrete,y_robot,u_robot)
stdev_robot_ukf_2 = np.sqrt(np.diagonal(P_robot_ukf_estimate_2,axis1=1,axis2=2))

f, ax = plt.subplots(1,3,figsize=(2*fig_size[0],fig_size[1]))
for i in range(3):
    ax[i].plot(t_robot,x_robot_ukf_estimate_2[:,i],linewidth=0.5, label='$\hat x_{}$ UKF'.format(i+1))
    ax[i].plot(t_robot,x_robot_ekf_estimate_2[:,i],linewidth=0.5, color='red', label='$\hat x_{}$ EKF'.format(i+1))
    ax[i].fill_between(t_robot,x_robot_ukf_estimate_2[:,i]-2*stdev_robot_ukf[:,i],x_robot_ukf_estimate[:,i]+2*stdev_robot_ukf[:,i],label='conf. UKF \n $x_{}$'.format(i+1),color='b',alpha=0.2)
    ax[i].fill_between(t_robot,x_robot_ekf_estimate_2[:,i]-2*stdev_robot_ekf[:,i],x_robot_ekf_estimate[:,i]+2*stdev_robot_ekf[:,i],label='conf. EKF \n $x_{}$'.format(i+1),color='r',alpha=0.2)
    ax[i].plot(t_robot,x_robot[:,i],'-k',linewidth=0.5, label='$x_{}$'.format(i+1))
    ax[i].set_xlabel('$t$')
    ax[i].set_ylabel('$x$')
    ax[i].legend()
plt.show()

f, ax = plt.subplots(2,2, figsize=(2*fig_size[0],2*fig_size[1]))
quiver_width=0.005
quiver_head_width=3.
quiver_head_length=4.
linewidth=1.

ax[0,0].plot(u_robot[:,0], label='Velocity', linewidth=linewidth)
ax[0,0].set_xlabel('$t$')
ax[0,0].set_ylabel('$v$')
ax[0,0].legend()
ax[0,1].plot(u_robot[:,1], label='Gyroscope', linewidth=linewidth)
ax[0,1].set_xlabel('$t$')
ax[0,1].set_ylabel('$\omega / \pi$')
ax[0,1].legend()

ax[1,0].plot(x_robot[:,0],x_robot[:,1], '-k', label='Robot-position-EM', linewidth=linewidth)
ax[1,0].quiver(x_robot[::skip,0],x_robot[::skip,1],
               np.cos(x_robot[::skip,2]),np.sin(x_robot[::skip,2]),
               label='Direction-EM', linewidth=linewidth, alpha=0.5, color='black',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(x_robot_ekf_estimate_2[:,0],x_robot_ekf_estimate_2[:,1], '-r', label='Robot-position-EKF', linewidth=linewidth)
ax[1,0].quiver(x_robot_ekf_estimate_2[::skip,0],x_robot_ekf_estimate_2[::skip,1],
               np.cos(x_robot_ekf_estimate_2[::skip,2]),np.sin(x_robot_ekf_estimate_2[::skip,2]),
               label='Direction-EKF', linewidth=linewidth, alpha=0.5, color='red',
                width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(x_robot_ukf_estimate[:,0],x_robot_ukf_estimate[:,1], '-b', label='Robot-position-UKF', linewidth=linewidth)
ax[1,0].quiver(x_robot_ukf_estimate_2[::skip,0],x_robot_ukf_estimate_2[::skip,1],
               np.cos(x_robot_ukf_estimate_2[::skip,2]),np.sin(x_robot_ukf_estimate_2[::skip,2]),
               label='Direction-UKF', linewidth=linewidth, alpha=0.5, color='blue',
               width=quiver_width, headwidth=quiver_head_width, headlength=quiver_head_length)
ax[1,0].plot(y_robot[:,0],y_robot[:,1], 'ok', markersize=1, label='Measurement', alpha=0.2)
ax[1,0].set_xlabel('$x_1$')
ax[1,0].set_ylabel('$x_2$')
ax[1,0].legend()
ax[1,1].plot(x_robot[:,2], '-k', label='Orientation-EM', linewidth=linewidth)
ax[1,1].plot(x_robot_ekf_estimate_2[:,2], '-r', label='Orientation-EKF', linewidth=linewidth)
ax[1,1].plot(x_robot_ukf_estimate_2[:,2], '-b', label='Orientation-UKF', linewidth=linewidth)
ax[1,1].set_xlabel('$t$')
ax[1,1].set_ylabel('$\phi / \pi$')
ax[1,1].legend()

plt.show()



