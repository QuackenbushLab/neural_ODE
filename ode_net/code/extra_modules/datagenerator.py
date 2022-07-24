import sys
import torch
import torch.nn as nn
import numpy as np
import math

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint
import random

def _testfunction(t, y):
    return -.3*y**3 + .3*y**2 + .2*y -.1

def _2d_testfunction(t, y):
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    return torch.mm(y**3, true_A)

def _1d_parabolic(t, y):
    g = 9.82
    grad = torch.tensor([y[0][1], -g])
    return grad

def _2d_parabolic(t, y):
    g = 9.82
    grad = torch.tensor([y[0][2], y[0][3], 0, -g])
    return grad

def _2d_parabolic_drag(t, y):
    g = 9.82
    k = 0.05
    m = 0.145
    vx = y[0][2]
    vy = y[0][3]
    ax = -k*vx*torch.abs(vx)/m
    ay = -g - k*vy*torch.abs(vy)/m
    grad = torch.tensor([vx, vy, ax, ay])                                                                                                      
    return grad

def _simple_harmonic(t, y):
    k = 2
    m = 1
    grad = torch.tensor([y[0][1], -k/m * y[0][0]])
    return grad

def _damped_harmonic(t, y):
    m = 1
    k = 2
    c = 0.5
    grad = torch.tensor([y[0][1], -k/m * y[0][0] - c/m * y[0][1]])
    return grad

def _lotka_volterra(t, y):
    a = 2/3
    b = 4/3
    g = 1
    d = 1
    grad = torch.tensor([y[0][0]*(a - b*y[0][1]), -y[0][1]*(g - d*y[0][0])])
    return grad

def _mystery_function(t, y):
    nfloor = int(y.shape[1] / 2)
    k = 0.001
    phase = 0.25
    F0 = .05
    
    masses = np.array([1000]*nfloor)
    couplings = np.array([k]*(nfloor+1))
    M = np.diag(masses)
    #print("M", M)


    #Couplings between floors
    K = []
    for floor in range(nfloor):
        floor_coupling = [0]*(nfloor+1)

        if floor>0:
            floor_coupling[floor-1] = couplings[floor]
        floor_coupling[floor] = -couplings[floor]-couplings[floor+1]
        floor_coupling[floor+1] = couplings[floor+1]

        K.append(floor_coupling)

    K = np.array(K)
    K = np.delete(K, -1, axis=1) # Remove last column
    #print("K", K)

    # External excitation (seismic wave)
    H = M
    H = -phase*phase* F0 * math.cos(phase*t)*H

    #print(H)

    #print(K.shape)
    #print(y.shape)
    _y = torch.zeros((y.shape[0], int(y.shape[1]/2)))
    for i in range(y.shape[1]):
        if i % 2 == 0:
            _y[:,int(i / 2)] = y[:,i]


    #_y = torch.stack([y[0][i*2] for i in range(int(y.shape[1] / 2))]).unsqueeze(0)
    Mxpp = np.add(np.multiply(K,_y),H)
    Mxpp = Mxpp.float()

    #print(Mxpp)
    tmp = torch.mm(_y, Mxpp)
    ret = torch.zeros(y.shape)
    for i in range(y.shape[1]):
        if i % 2 == 0:
            ret[0][i] = y[0][i+1]
        else:
            ret[0][i] = tmp[0][int(i/2)]

    return ret

def _mystery_function_2_floors(t, y):
    nfloor = int(y.shape[1] / 2)
    k = 10000
    phase = 0.5
    F0 = 2
    m = 1000

    x_dot = k/m * (-2*y[:,0] + y[:,2]) - F0*phase*phase*math.cos(phase * t)
    y_dot = k/m * (y[:,0] - y[:,2]) - F0*phase*phase*math.cos(phase * t)

    grad = torch.tensor([y[:,1], x_dot, y[:,3], y_dot])
    return grad
    

class DataGenerator(nn.Module):

    def __init__(self, ntraj, y0_range, t_range, num_times, noise_scale, device, function="test", method="dopri5", random_y0=False):
        super(DataGenerator, self).__init__()
        self.ntraj = ntraj
        self.device = device
        self.method = method
        self.noise_scale=noise_scale

        if function == "test":
            self.function = _testfunction
        elif function == "2d_test":
            self.function = _2d_testfunction
        elif function == "1d_parabolic":
            self.function = _1d_parabolic
        elif function == "2d_parabolic":
            self.function = _2d_parabolic
        elif function == "2d_parabolic_drag":
            self.function = _2d_parabolic_drag
        elif function == "simple_harmonic":
            self.function = _simple_harmonic
        elif function == "damped_harmonic":
            self.function = _damped_harmonic
        elif function == "mystery_function":
            self.function = _mystery_function_2_floors
        elif function == "lotka_volterra":
            self.function = _lotka_volterra
        else:
            raise ValueError

        # Generate y0 in the specified interval
        self.dim = int(len(y0_range)/2)
        self.y0 = np.zeros((ntraj, self.dim), dtype=np.float32)
        if random_y0:
            for i in range(self.dim):
                self.y0[:, i] = np.random.uniform(low=y0_range[i*2], high=y0_range[i*2+1], size=ntraj)
        else:
            for i in range(self.dim):
                self.y0[:, i] = np.linspace(y0_range[i*2], y0_range[i*2+1], num=ntraj)        
        self.y0 = torch.from_numpy(self.y0).reshape((ntraj, 1, self.dim)).to(self.device)

        # Generate t for each trajectory
        self.generate_t(ntraj, t_range, num_times)

    def add_noise_to_traj(self, size, traj_index):
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=size)
        noise = np.float32(noise)
        self.data_np[traj_index] = self.data_np[traj_index] + noise
        self.data_pt[traj_index] = self.data_pt[traj_index] + torch.from_numpy(noise).to(self.device)

    def generate(self):
        ''' Generate the data and return it as a np-vector and a pytorch-tensor '''
        self.data_pt = []
        self.data_np = []
        with torch.no_grad():
            for i in range(self.ntraj):
                data = odeint(self, self.y0[i], self.t_pt[i], method=self.method).to(self.device)
                self.data_pt.append(data)
                self.data_np.append(data.numpy())

                # Generate noise and add it to the data
                if self.noise_scale:
                    self.add_noise_to_traj(data.shape, i)

        return self.data_np, self.data_pt, self.t_np, self.t_pt

    def forward(self, t, y):
        ''' Forward prop the defined function '''
        return self.function(t, y)

    def generate_t(self, ntraj, t_range, num_times):
        ''' Generate the time series corresponding to the data, can be changed later '''
        t_mat = []
        t = np.linspace(0, t_range, num_times)
        for i in range(ntraj):
            t_mat.append(torch.from_numpy(t).to(self.device))
        self.t_pt = torch.stack(t_mat).to(self.device)
        self.t_np = []
        for traj in self.t_pt:
            self.t_np.append(traj.numpy())