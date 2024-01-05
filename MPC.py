import time
from f110_gym.envs.base_classes import Integrator
import gym
import yaml
import numpy as np
from argparse import Namespace
import casadi as ca
import math, cmath
import collections
import matplotlib.pyplot as plt
from DataSave import dataSave

mu = 0.
sigma = 0.2

class MPCPlanner:
    def __init__(self,param, dt, N , map_name, TESTMODE):
        self.dt = dt
        self.N = N
        self.map_name = map_name
        self.TESTMODE = TESTMODE
        self.reset = 0

        self.scale = 0.

        self.L = 0.324
        self.nx = 4
        self.nu = 2
        self.u_min = [-0.4, -13]
        self.u_max = [0.4, 13]

        if self.TESTMODE == "Benchmark" or self.TESTMODE == " ":
            self.dt_gain = param.Benchmark_dt_gain                #change this parameter for different tracks 
            self.dt_constant = param.Benchmark_dt_constant      
            # self.Max_iter = param.Benchmark_Max_iter    
            self.Max_iter=7           
        elif self.TESTMODE == "perception_noise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
            self.dt_gain = param.noise_dt_gain                #change this parameter for different tracks 
            self.dt_constant = param.noise_dt_constant                     #lood forward distance constant
            self.Max_iter = param.noise_Max_iter
        elif self.TESTMODE == "dt_gain":
            self.dt_gain = param.gain_tune_dt_gain                #change this parameter for different tracks 
            self.dt_constant = param.gain_tune_dt_constant                     #lood forward distance constant
            self.Max_iter = param.gain_tune_Max_iter
        elif self.TESTMODE == "dt_constant":
            self.dt_gain = param.lfd_tune_dt_gain                #change this parameter for different tracks 
            self.dt_constant = param.lfd_tune_dt_constant                     #lood forward distance constant
            self.Max_iter = param.lfd_tune_Max_iter
        elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_Delay_steering" or self.TESTMODE == "perception_delay":
            self.dt_gain = param.delay_dt_gain                #change this parameter for different tracks 
            self.dt_constant = param.delay_dt_constant                     #lood forward distance constant
            self.Max_iter = param.delay_Max_iter

        self.ds = dataSave(TESTMODE, map_name, self.Max_iter)
        self.waypoints = np.loadtxt('./maps/'+self.map_name+'_raceline.csv', delimiter=",")
        # print(self.dt_constant,self.dt_gain)


    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        self.drawn_waypoints = []
        self.wpts = np.vstack((self.waypoints[:,1],self.waypoints[:,2])).T
        scaled_points = 50.*self.wpts

        for i in range(scaled_points.shape[0]):
            if len(self.drawn_waypoints) < scaled_points.shape[0]:
                b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        self.trueSpeedProfile = self.waypoints[:, 5]
        self.vs = self.trueSpeedProfile   #speed profile

        self.total_s = self.ss[-1]
        self.tN = len(self.wpts)

    def get_timed_trajectory_segment(self, position, dt, n_pts=10):
        pose = np.array([position[0], position[1], position[3]])
        trajectory, distances = [pose], [0]
        for i in range(n_pts-1):
            distance = dt * pose[2]
            
            current_distance = self.calculate_progress(pose[0:2])
            next_distance = current_distance + distance
            distances.append(next_distance)
            
            interpolated_x = np.interp(next_distance, self.ss, self.wpts[:, 0])
            interpolated_y = np.interp(next_distance, self.ss, self.wpts[:, 1])

            interpolated_v = np.interp(next_distance, self.ss, self.vs)
            pose = np.array([interpolated_x, interpolated_y, interpolated_v])
            trajectory.append(pose)
        
        interpolated_waypoints = np.array(trajectory)
        # print(interpolated_waypoints)
        return interpolated_waypoints


    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.ss[idx+1] - self.ss[idx]

        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                # negative due to floating point precision
                # if the point is very close to the trackline, then the trianlge area is increadibly small
                h = 0
                x = d_ss + d1
                # print(f"Area square is negative: {Area_square}")
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h
    
    def get_trackline_segment(self, point):
        """
        Returns the first index representing the line segment that is closest to the point.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)

        if min_dist_segment == len(self.wpts)-1:
            min_dist_segment = 0

        return min_dist_segment,dists
        
    def calculate_progress(self, point):
        idx, dists = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, dists)
        s = self.ss[idx] + x
        
        return s
    

    def estimate_u0(self, reference_path, x0):

        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])

        th_dot = calculate_angle_diff(reference_theta) 

        th_dot[0] += (reference_theta[0]- x0[2]) 

        
        speeds = reference_path[:, 2]
        steering_angles = (np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt

        speeds[0] += (x0[3] - reference_path[0, 2])
        accelerations = np.diff(speeds) / self.dt
        
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        # print("estimate steering",steering_angles[0])
        
        return u0_estimated
        
    def plan(self, obs, laptime):
        x0 = self.inputStateAdust(obs)
        self.dt = self.dt_gain * x0[3] + self.dt_constant
        reference_path = self.get_timed_trajectory_segment(x0, self.dt, self.N+2)
        u0_estimated = self.estimate_u0(reference_path, x0)

        #figure out the reference path 
        u_bar, x_bar = self.generate_optimal_path(x0, reference_path[:-1].T, u0_estimated)
        # self.realTimePlot(x_bar)

        speed = x0[3] + u_bar[0][1]*self.dt
        speed,steering = self.outputActionAdjust(speed,u_bar[0][0])
        # print(self.dt)
        pose = np.array([x0[0], x0[1]])
        ego_index,min_dists = self.get_trackline_segment(pose)
        self.completion = 100 if ego_index/len(self.wpts) == 0 else round(ego_index/len(self.wpts)*100,2)
        _,trackErr = self.interp_pts(ego_index, min_dists)
        self.ds.saveStates(laptime,x0,speed,trackErr,self.scaledRand)
        # print("u_bar",u_bar[0][0], speed)


        return speed, steering  # return the first control action
        
    def generate_optimal_path(self, x0_in, x_ref, u_init):
        """generates a set of optimal control inputs (and resulting states) for an initial position, reference trajectory and estimated control

        Args:
            x0_in (ndarray(3)): the initial pose
            x_ref (ndarray(N+1, 2)): the reference trajectory
            u_init (ndarray(N)): the estimated control inputs

        Returns:
            u_bar (ndarray(N)): optimal control plan
        """
        x = ca.SX.sym('x', self.nx, self.N+1)
        u = ca.SX.sym('u', self.nu, self.N)
        
        speeds = x_ref[2]

        # Add a speed objective cost.
        J = ca.sumsqr(x[:2, :] - x_ref[:2, :])  + ca.sumsqr(x[3, :] - speeds[None, :]) *10
        g = []
        for k in range(self.N):
            x_next = x[:,k] + self.f(x[:,k], u[:,k])*self.dt
            g.append(x_next - x[:,k+1])

        initial_constraint = x[:,0] - x0_in 
        g.append(initial_constraint)

        
        x_init = [x0_in]

        for i in range(1, self.N+1):
            x_init.append(x_init[i-1] + self.f(x_init[i-1], u_init[i-1])*self.dt)

        for i in range(len(u_init)):
            x_init.append(u_init[i])



        x_init = ca.vertcat(*x_init)
        # print("x_init.shape : ",x_init.shape)

        
        lbx = [-ca.inf, -ca.inf, -ca.inf, 0] * (self.N+1) + self.u_min * self.N
        ubx = [ca.inf, ca.inf, ca.inf, 8] * (self.N+1) + self.u_max * self.N
        # print ("lbx = ",lbx)
        # print("ubx = ", ubx)
 
        x_nlp = ca.vertcat(x.reshape((-1, 1)), u.reshape((-1, 1)))
        # print("x_nlp: ", x_nlp,"shape", x_nlp.shape)
        g_nlp = ca.vertcat(*g)
        # print("g_nlp: ", g_nlp,"shape", g_nlp.shape)

        nlp = {'x': x_nlp,
            'f': J,
            'g': g_nlp}
        
        opts = {'ipopt': {'print_level': 2},
                'print_time': False}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0)
        # print(sol['x'])
        x_bar = np.array(sol['x'][:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        # print(x_bar)

        u_bar = sol['x'][self.nx*(self.N+1):]

        u_return = np.array(u_bar)[:, 0]
        u_return = u_return.reshape((self.N, self.nu))
        
        return u_return, x_bar
    
    def f(self, x, u):
        # define the dynamics as a casadi array
        xdot = ca.vertcat(
            ca.cos(x[2])*x[3],
            ca.sin(x[2])*x[3],
            x[3]/self.L * ca.tan(u[0]),
            u[1]
        )
        return xdot
    
    def inputStateAdust(self,obs):
        if self.reset:
            X0 = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], 8.]
            self.reset = 0
        else:
            X0 = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0]]


        if self.TESTMODE == "perception_noise":
            rand = np.random.normal(mu,sigma,1)
            self.scaledRand = rand*self.scale
            X0 = [obs['poses_x'][0]+self.scaledRand[0], obs['poses_y'][0]+self.scaledRand[0], obs['poses_theta'][0], obs['linear_vels_x'][0]+self.scaledRand[0]]

        return X0
    
    def outputActionAdjust(self,speed,steering):
        rand = np.random.normal(mu,sigma,1)
        self.scaledRand = rand*self.scale

        speed_mod = speed
        steering_mod = steering

        if self.TESTMODE == "Outputnoise_speed":
            speed_mod = speed + self.scaledRand[0]
        elif self.TESTMODE == "Outputnoise_steering":
            steering_mod = steering + self.scaledRand[0]

        return speed_mod, steering_mod
    
    def realTimePlot(self,x_bar):
        plt.plot(self.wpts[:,0],self.wpts[:,1],"bx",markersize=1)
        plt.plot(x_bar[0, :], x_bar[1, :], 'bo', markersize=4, label="Solution States (x_bar)") 
        plt.pause(0.001)
        plt.clf()
        
    
#................................................................................

def calculate_angle_diff(angle_vec):
    angle_diff = np.zeros(len(angle_vec)-1)
    
    for i in range(len(angle_vec)-1):
        angle_diff[i] = sub_angles_complex(angle_vec[i], angle_vec[i+1])
    
    return angle_diff
    
def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

