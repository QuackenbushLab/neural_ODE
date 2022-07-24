import torch
import numpy as np
from odenet import ODENet
from datahandler import DataHandler

def _2d_parabolic(t, y):
    g = 9.82
    grad = torch.tensor([y[2], y[3], 0, -g])
    return grad

def print_A_nice(A, order, left_hand):
    variables = A.shape[1]
    variables = left_hand.copy()
    for i in range(2, order+1):
        variables += [''.join([variable, '^{}'.format(i)]) for variable in left_hand]
    variables.append('')
    for lh, coeffs in zip(left_hand, A):
        rh = ' '.join('{:+.4f}{}'.format(coeff, var) for coeff, var in zip(coeffs, variables))
        print('{} = {}'.format(lh, rh))

def generate_grid(gridsize, grid_range, dim):
    data_ranges = []
    for i in range(dim):
        r = np.linspace(grid_range[i*2], grid_range[i*2 + 1], gridsize)
        data_ranges.append(r)
    
    x_meshgrid = np.meshgrid(*data_ranges)
    x_np = np.vstack([array for array in map(np.ravel, x_meshgrid)])
    x_np = np.transpose(x_np)
    x_torch = torch.from_numpy(x_np).float()
    y = torch.ones((gridsize**dim, dim + 1))
    grad = torch.zeros((gridsize**dim, dim))
    return x_torch, y, grad, x_meshgrid

def solve_eq(odenet, gridsize=None, grid_range=None, data=None, order=1):
    '''
    Find the matrix A and vector b which describe the ODE dx/dt = A[x, x^2, ..., x^n] + b
    using lstsq. Returns a [dim, dim + 1] array where the last column is
    the b-vector.
    '''
    with torch.no_grad():
        dim = odenet.ndim
        if not data:
            x_torch, y, grad = generate_grid(gridsize, grid_range, dim)
        else:
            x_torch = torch.squeeze(torch.cat(data), dim=1)
            y = torch.ones((x_torch.shape[0], dim + 1))
            grad = torch.zeros((x_torch.shape[0], dim))
        t = torch.zeros(1)

        y[:,0:dim] = x_torch
        grad[:,:] = odenet.forward(t, x_torch)

        y_np = y.numpy()
        if order > 1:
            y = np.ones((y_np.shape[0], dim*order + 1))
            y[:, 0:dim] = y_np[:, 0:dim]
            for i in range(2, order + 1):
                y[:, (i - 1)*dim:i*dim] = y[:, 0:dim]*np.abs(np.power(y[:, 0:dim], i-1))
        else:
            y = y_np
        grad_np = grad.detach().numpy()

        
        a = []
        for i in range(dim):
            a_ = np.linalg.lstsq(y, grad_np[:, i], rcond=None)
            a.append(a_[0])

        return np.array(a)

if __name__ == "__main__":
    dim = 4
    data_file = 'data/2d_parabolic_drag_random_x_and_y.csv'
    model_file = '/home/daniel/code/Msc-Thesis/fully_trained/2019-3-19(14;11)_2d_parabolic_drag_random_x_and_y_40epochs/final_model_40epochs.pt'
    order = 2
    left_hand = ['x', 'y', 'x_dot', 'y_dot']
    data_handler = DataHandler.fromcsv(data_file, 'cpu', 0.0)

    data = data_handler.data_pt
    data_points = 50
    span = (-3, 3, 10, 0, -3, 3, 0, -10)

    odenet = ODENet('cpu', dim)
    odenet.load(model_file)
    A = solve_eq(odenet, data_points, span, data, order=order)
    
    print_A_nice(A, order, left_hand)
