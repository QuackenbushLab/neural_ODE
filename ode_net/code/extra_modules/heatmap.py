import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
from solve_eq import solve_eq
from odenet import ODENet
from solve_eq import generate_grid
from datahandler import DataHandler
from figure_saver import save_figure, set_font_settings
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint


def _1d_parabolic(t, y):
    g = 9.82
    grad = torch.cat((y[:,1].unsqueeze(dim=1), -g*torch.ones(y.shape[0], 1)), dim=1)
    return grad

def _2d_parabolic(t, y):
    g = 9.82
    grad = torch.cat((y[:,2].unsqueeze(dim=1), y[:,3].unsqueeze(dim=1), torch.zeros(y.shape[0], 1),  -g*torch.ones(y.shape[0], 1)), dim=1)
    return grad

def _simple_harmonic(t, y):
    constant = 2
    true_A = torch.tensor([[0.0, -constant], [1.0, 0.0]])
    return torch.mm(y, true_A)

def _damped_harmonic(t, y):
    m = 1
    k = 2
    c = 0.5
    true_A = torch.tensor([[0.0, -k/m], [1.0, -c/m]])
    return torch.mm(y, true_A)

def _2d_parabolic_drag(t, y):
    g = 9.82
    k = 0.05
    m = 0.145
    vx = y[:,2]
    vy = y[:,3]
    ax = -k*vx*torch.abs(vx)/m
    ay = -g - k*vy*torch.abs(vy)/m
    grad = torch.cat((vx.unsqueeze(dim=1), vy.unsqueeze(dim=1), ax.unsqueeze(dim=1), ay.unsqueeze(dim=1)), dim=1)                                                                                                    
    return grad

def _lotka_volterra(t, y):
    a = 2/3
    b = 4/3
    g = 1
    d = 1
    a1 = y[:,0]*(a - b*y[:,1])
    a2 = -y[:,1]*(g - d*y[:,0])
    grad = torch.cat((a1.unsqueeze(dim=1), a2.unsqueeze(dim=1)), dim=1)
    return grad

def plot_error_distribution(true_grad, linear_grad, network_grad, filename):
    fig_hist_linear, ax_linear = plt.subplots(ncols=1)
    fig_hist_network, ax_network = plt.subplots(ncols=1)

    err_linear  = torch.norm(true_grad - linear_grad, dim=1)/torch.norm(true_grad, dim=1)
    err_network = torch.norm(true_grad - network_grad, dim=1)/torch.norm(true_grad, dim=1)
    ax_linear.hist(err_linear, bins=200, weights=np.ones_like(err_linear)/err_linear.shape[0])
    ax_linear.set_ylabel("Frequency")
    ax_linear.set_xlabel("Relative error")
    ax_network.hist(err_network, bins=200, weights=np.ones_like(err_network)/err_network.shape[0]) 
    ax_network.set_ylabel("Frequency")
    ax_network.set_xlabel("Relative error")
    fig_hist_linear.canvas.set_window_title('Relative error linear approx histogram')
    fig_hist_network.canvas.set_window_title('Relative error network approx histogram')
    save_figure(fig_hist_linear, "../images/hist_linear_error_{}.eps".format(filename.replace(".csv", "")), width=5.9/2)
    save_figure(fig_hist_network, "../images/hist_network_error_{}.eps".format(filename.replace(".csv", "")), width=5.9/2)

def make_1d_quiver_plots(x_mesh, true_grad, linear_grad, network_grad):
    fig_dyn, (ax_real, ax_linear, ax_network) = plt.subplots(ncols=3)
    fig_all, ax = plt.subplots(ncols=1)

    true_grad_x = true_grad[:, 0].reshape(x_mesh[0].shape)
    true_grad_y = true_grad[:, 1].reshape(x_mesh[1].shape)
    linear_grad_x = linear_grad[:, 0].reshape(x_mesh[0].shape)
    linear_grad_y = linear_grad[:, 1].reshape(x_mesh[1].shape)
    network_grad_x = network_grad[:, 0].reshape(x_mesh[0].shape)
    network_grad_y = network_grad[:, 1].reshape(x_mesh[1].shape)
    ax_real.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], true_grad_x[0::4,0::4], true_grad_y[0::4,0::4])
    ax_linear.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], linear_grad_x[0::4,0::4], linear_grad_y[0::4,0::4])
    ax_network.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], network_grad_x[0::4,0::4], network_grad_y[0::4,0::4])

     
    ax_real.set_title("Real")
    ax_network.set_title("Network")
    ax_linear.set_title("Linear")

    real = ax.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], true_grad_x[0::4,0::4], true_grad_y[0::4,0::4], color='black', label='Real derivative')
    linear = ax.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], linear_grad_x[0::4,0::4], linear_grad_y[0::4,0::4], color='red', label='Linear derivative')
    network = ax.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], network_grad_x[0::4,0::4], network_grad_y[0::4,0::4], color='blue', label='Network derivative')
    fig_all.legend()

def make_2d_heat_plots(data, x_torch, x_mesh, grad1, grad2, label1, label2, filename, axislabels):
    fig_rel_heat, ax_rel_heat = plt.subplots(ncols=1)

    grad_diff = grad1 - grad2
    grad_abs = torch.norm(grad_diff, dim=1)
    grad1_abs = torch.norm(grad1, dim=1)
    grad_rel = grad_abs/grad1_abs

    x_torch = x_torch.reshape((gridsize, gridsize, gridsize, gridsize, 4))
    grad_rel = grad_rel.reshape((gridsize, gridsize, gridsize, gridsize))
    grad_abs = grad_abs.reshape((gridsize, gridsize, gridsize, gridsize))
    
    grad_rel = torch.mean(torch.mean(grad_rel, dim=2), dim=2)
    
    grad_abs = torch.mean(torch.mean(grad_abs, dim=2), dim=2)

    im_rel = ax_rel_heat.imshow(grad_rel, extent=grid_range[0:4], origin='lower', cmap='inferno')
    divider = make_axes_locatable(ax_rel_heat)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig_rel_heat.colorbar(im_rel, cax=cax, shrink=0.9)

    for idx, traj in enumerate(data):
        ax_rel_heat.plot(traj[:,:,0].flatten(), traj[:,:,1].flatten(), '-r', linewidth=3, alpha=0.7)
    
    trainig_data_legend = mpatches.Patch(color='red', label='Training data')

    grad1_x = grad1[:, 0].reshape(x_mesh[0].shape)
    grad1_y = grad1[:, 1].reshape(x_mesh[1].shape)
    grad2_x = grad2[:, 0].reshape(x_mesh[0].shape)
    grad2_y = grad2[:, 1].reshape(x_mesh[1].shape)

    ax_rel_heat.set_xlabel(axislabels[0])
    ax_rel_heat.set_ylabel(axislabels[1])


    #real = ax_rel_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad1_x[0::4,0::4], grad1_y[0::4,0::4], color='lime', label=label1)
    #approx = ax_rel_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad2_x[0::4,0::4], grad2_y[0::4,0::4], color='cyan', label=label2)
    fig_rel_heat.legend(handles=[trainig_data_legend], mode='expand', loc='upper center', ncol=3)

    fig_rel_heat.canvas.set_window_title('Relative error heatmap')
    save_figure(fig_rel_heat, '../images/rel_heat_{}_{}_{}.eps'.format(filename.replace(".csv", ""), label1.replace(" ", "_"), label2.replace(" ", "_")), square=True)

def make_1d_heat_plots(data, x_torch, x_mesh, grad1, grad2, label1, label2, filename, axislabels):

    #fig_heat, ax_heat = plt.subplots(ncols=1)
    fig_rel_heat, ax_rel_heat = plt.subplots(ncols=1)

    grad_diff = grad1 - grad2
    grad_abs = torch.norm(grad_diff, dim=1)
    grad1_abs = torch.norm(grad1, dim=1)
    grad_rel = grad_abs/grad1_abs

    x_torch = x_torch.reshape((gridsize, gridsize, 2))
    grad_rel = grad_rel.reshape((gridsize, gridsize))
    grad_abs = grad_abs.reshape((gridsize, gridsize))

    #im = ax_heat.imshow(grad_abs, extent=grid_range, origin='lower', cmap='inferno')
    #fig_heat.colorbar(im, ax=ax_heat, shrink=0.9)
    im_rel = ax_rel_heat.imshow(grad_rel, extent=grid_range, origin='lower', cmap='inferno')
    divider = make_axes_locatable(ax_rel_heat)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig_rel_heat.colorbar(im_rel, cax=cax, shrink=0.9)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Relative error', rotation=270)

    for idx, traj in enumerate(data):
        #ax_heat.plot(traj[:,:,0].flatten(), traj[:,:,1].flatten(), '-r', linewidth=3, alpha=0.7)
        ax_rel_heat.plot(traj[:,:,0].flatten(), traj[:,:,1].flatten(), '-r', linewidth=3, alpha=0.7)
    
    trainig_data_legend = mpatches.Patch(color='red', label='Training data')

    grad1_x = grad1[:, 0].reshape(x_mesh[0].shape)
    grad1_y = grad1[:, 1].reshape(x_mesh[1].shape)
    grad2_x = grad2[:, 0].reshape(x_mesh[0].shape)
    grad2_y = grad2[:, 1].reshape(x_mesh[1].shape)
   
    #real = ax_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad1_x[0::4,0::4], grad1_y[0::4,0::4], color='lime', label=label1)
    #approx = ax_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad2_x[0::4,0::4], grad2_y[0::4,0::4], color='cyan', label=label2)
    #fig_heat.legend(handles=[trainig_data_legend, real, approx], loc='upper center', ncol=3)

    ax_rel_heat.set_xlabel(axislabels[0])
    ax_rel_heat.set_ylabel(axislabels[1])


    real = ax_rel_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad1_x[0::4,0::4], grad1_y[0::4,0::4], color='lime', label=label1)
    approx = ax_rel_heat.quiver(x_mesh[0][0::4,0::4], x_mesh[1][0::4,0::4], grad2_x[0::4,0::4], grad2_y[0::4,0::4], color='cyan', label=label2)
    fig_rel_heat.legend(handles=[trainig_data_legend, real, approx], mode='expand', loc='upper center', ncol=3)

    #fig_heat.canvas.set_window_title('Absolute error heatmap')
    fig_rel_heat.canvas.set_window_title('Relative error heatmap')

    #save_figure(fig_heat, '../images/abs_heat_{}_{}.eps'.format(label1.replace(" ", "_"), label2.replace(" ", "_")), square=True)
    save_figure(fig_rel_heat, '../images/rel_heat_{}_{}_{}.eps'.format(filename.replace(".csv", ""), label1.replace(" ", "_"), label2.replace(" ", "_")), square=True)

def pad_x(x_torch, order):
    if order > 1:
        exponent = 2
        last_factor = torch.mul(x_torch, torch.abs(x_torch))
        padded_x = torch.cat((x_torch, last_factor), dim=1)
        while exponent < order:
            last_factor = torch.mul(x_torch, torch.abs(x_torch))
            padded_x = torch.cat((padded_x, last_factor), dim=1)
            exponent += 1

    else:
        padded_x = x_torch
    
    return torch.cat((padded_x, torch.ones(x_torch.shape[0], 1)), dim=1)

if __name__ == '__main__':
    set_font_settings()
    parser = argparse.ArgumentParser('Parser')
    #parser.add_argument('--file', type=str, default='D:\\Skola\\MSc-Thesis\\Base\\output\\2019-3-19(15;58)_damped_harmonic_20epochs\\best_model.pt')
    #parser.add_argument('--file', type=str, default='/home/daniel/code/Msc-Thesis/fully_trained/2019-4-5(14;6)_1d_parabolic_rnd_vel2_100epochs/best_model.pt')
    parser.add_argument('--file', type=str, default='/home/daniel/code/Msc-Thesis/fully_trained/2019-3-13(14;8)_2d_parabolic_40epochs/final_model_40epochs.pt')
    args = parser.parse_args()

    filename = '2d_parabolic.csv'
    real_ode = _2d_parabolic

    dh = DataHandler.fromcsv('data/{}'.format(filename), device='cpu', val_split=0.0)

    data = dh.data_np
    times = dh.time_np
    dim = data[0].shape[2]
    order = 1
    odenet = ODENet('cpu', dim)
    odenet.load(args.file)
    A = solve_eq(odenet, data=dh.data_pt, order=order)
    A = torch.tensor(A).float()
    print(A)

    gridsize = 60
    grid_range = (-30, 30, -30, 30, -30, 30, -30, 30,)
    axislabels = (r'$x$ [m]', r'$\frac{\mathrm{d}x}{\mathrm{d}t}$ [ms$^{-1}$]')

    with torch.no_grad():
        print(odenet.ndim)
        x_torch, y, network_grad, x_mesh = generate_grid(gridsize, grid_range, odenet.ndim)
        t = torch.zeros(1)
        print("Doing forward")
        network_grad[:,:] = odenet.forward(t, x_torch)
        print("Forward done")
        true_grad = real_ode(t, x_torch)
        print("Real ODE done")
        padded_x = pad_x(x_torch, order)
        linear_grad = torch.mm(padded_x, torch.transpose(A, 1, 0))
        print("Real grad done")
        if dim == 2:
            #make_1d_quiver_plots(x_mesh, true_grad, linear_grad, network_grad)
            make_1d_heat_plots(data, x_torch, x_mesh, true_grad, network_grad, 'True derivative', 'Network derivative', filename, axislabels=axislabels)
            make_1d_heat_plots(data, x_torch, x_mesh, true_grad, linear_grad, 'True derivative', 'Linearized derivative', filename, axislabels=axislabels)
            #make_1d_heat_plots(data, x_torch, x_mesh, network_grad, linear_grad, 'Network derivative', 'Linearized derivative')
            plot_error_distribution(true_grad, linear_grad, network_grad, filename)
        elif dim == 4:
            plot_error_distribution(true_grad, linear_grad, network_grad, filename)
            #make_2d_heat_plots(data, x_torch, x_mesh, true_grad, network_grad, 'True derivative', 'Network derivative', filename, axislabels=axislabels)
            #make_2d_heat_plots(data, x_torch, x_mesh, true_grad, linear_grad, 'True derivative', 'Linearized derivative', filename, axislabels=axislabels)
        else:
            raise ValueError("This seems wrong")
        print("Images should be saved in ../images")
        print("Showing images")
        plt.show()
        