import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from figure_saver import save_figure
import numpy as np
import torch
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint


class Visualizator():

    def visualize(self):
        pass

    def __init__(self, data_handler, odenet, settings):
        self.data_handler = data_handler
        self.odenet = odenet
        self.settings = settings

    def save_plot(self, fig, folder, name):
        fig.savefig('{}/{}.eps'.format(folder, name))

class Visualizator1D(Visualizator):
    
    def __init__(self, data_handler, odenet, settings):
        super().__init__(data_handler, odenet, settings)
        # IH: uncomment this when vis dyn
        #self.fig_dyn = plt.figure(figsize=(6,6))
        #self.fig_dyn.canvas.set_window_title("Dynamics")
        #self.ax_dyn = self.fig_dyn.add_subplot(111, frameon=False)
        
        self.fig_traj_split = plt.figure(figsize=(15,15), tight_layout=True)
        self.fig_traj_split.canvas.set_window_title("Trajectories in each dimension")
        self.TOT_ROWS = 5
        self.TOT_COLS = 6
        self.genes_to_viz = [1,6,14,16,21,32,37,39,41,43,46,48,49,56,58,60,75,85,92,99,101,103,115,118,120,122,127,128,142,143]
        self.axes_traj_split = self.fig_traj_split.subplots(nrows=self.TOT_ROWS, ncols=self.TOT_COLS, sharex=False, sharey=True, subplot_kw={'frameon':True})
        self.legend_traj = [Line2D([0], [0], color='black', linestyle='-.', label='NN approx. of dynamics'),Line2D([0], [0], color='green', linestyle='-', label='True dynamics'),Line2D([0], [0], marker='o', color='red', label='Observed data', markerfacecolor='red', markersize=5)]

        self.fig_traj_split.legend(handles=self.legend_traj, loc='upper center', ncol=3)
        
        self._set_ax_limits()

        #plt.show()
        #plt.savefig('initial_plot.png')
        
    def plot(self):
        #plt.figure(1)
        # IH: uncomment this when vis dyn
        #self.fig_dyn.canvas.draw_idle()
        #self.fig_dyn.canvas.start_event_loop(0.005)
        #plt.figure(2)
        self.fig_traj_split.canvas.draw_idle()
        self.fig_traj_split.canvas.start_event_loop(0.005)

    def _set_ax_limits(self):
        data = self.data_handler.data_np
        times = self.data_handler.time_np
        self.EXTRA_WIDTH_TRAJ = 0.2
        self.EXTRA_WIDTH_DYN = 1

        #self.x_span = (np.min([np.min(traj[:,:,0]) for traj in data]),
        #               np.max([np.max(traj[:,:,0]) for traj in data]))
        #self.x_width = self.x_span[1] - self.x_span[0]

        #self.xdot_span = (np.min([np.min(traj[:,:,1]) for traj in data]),
        #                  np.max([np.max(traj[:,:,1]) for traj in data]))
        #self.xdot_width = self.xdot_span[1] - self.xdot_span[0]

        self.time_span = (np.min([np.min(time[:]) for time in times]),
                          np.max([np.max(time[:]) for time in times]))
        self.time_width = self.time_span[1] - self.time_span[0]

    
        # IH: uncomment this when vis dyn
        #self.ax_dyn.set_xlim((self.x_span[0]-self.x_width*self.EXTRA_WIDTH_DYN,
        #                       self.x_span[1]+self.x_width*self.EXTRA_WIDTH_DYN))
        #self.ax_dyn.set_ylim((self.xdot_span[0]-self.xdot_width*self.EXTRA_WIDTH_DYN,
        #                       self.xdot_span[1]+self.xdot_width*self.EXTRA_WIDTH_DYN))

        for row_num,this_row_plots in enumerate(self.axes_traj_split):
            for col_num, ax in enumerate(this_row_plots):
                ax.set_xlim((self.time_span[0]-self.time_width*self.EXTRA_WIDTH_TRAJ,
                            self.time_span[1]+self.time_width*self.EXTRA_WIDTH_TRAJ))
                ax.set_ylim((-0.2,1.2))
         

    def visualize(self):
        self.trajectories = self.data_handler.calculate_trajectory(self.odenet, self.settings['method'])
        self._visualize_trajectories_split()
        #self._visualize_dynamics()
        self._set_ax_limits()

    def _visualize_trajectories_split(self):
        times = self.data_handler.time_np
        for row_num,this_row_plots in enumerate(self.axes_traj_split):
            for col_num, ax in enumerate(this_row_plots):
                gene = self.genes_to_viz[row_num*self.TOT_COLS + col_num] #IH restricting to plot only up to 30 genes
                ax.cla()
                for sample_num, (approx_traj, traj, true_mean) in enumerate(zip(self.trajectories, self.data_handler.data_np, self.data_handler.data_np_0noise)):
                    if sample_num > 7: #IH restrciting to plotting only 7 genes
                        break 
                    ax.plot(times[sample_num].flatten(), traj[:,:,gene].flatten(), 'r-o', alpha=0.15)
                    ax.plot(times[sample_num].flatten(), true_mean[:,:,gene].flatten(),'g-', lw=1.5)
                    ax.plot(times[sample_num].flatten(), approx_traj[:,:,gene].numpy().flatten(),'k-.', lw=1)
                
                ax.set_xlabel(r'$t$')
        
            
    def _visualize_dynamics(self):
        GRIDSIZE = 20

        self.ax_dyn.cla()
        for j in range(self.data_handler.ntraj-1, -1, -1):
            self.ax_dyn.plot(self.data_handler.data_np[j][:,:,0].flatten(), self.data_handler.data_np[j][:,:,1].flatten(), '-r', alpha=0.3, linewidth=2)

        xv, yv = torch.meshgrid([torch.linspace(self.x_span[0]-self.x_width*self.EXTRA_WIDTH_DYN, 
                                                self.x_span[1]+self.x_width*self.EXTRA_WIDTH_DYN, 
                                                GRIDSIZE),
                                 torch.linspace(self.xdot_span[0]-self.xdot_width*self.EXTRA_WIDTH_DYN,
                                                self.xdot_span[1]+self.xdot_width*self.EXTRA_WIDTH_DYN, 
                                                GRIDSIZE)])
        inputs = torch.from_numpy(np.vstack([array for array in map(np.ravel, (xv, yv))])).to(self.data_handler.device)
        inputs = torch.transpose(inputs, 1, 0)
        t = torch.zeros(1).to(self.data_handler.device)
        grad = self.odenet.forward(t, inputs)
        grad_x = grad[:, 0].reshape(xv.shape).cpu().numpy()
        grad_y = grad[:, 1].reshape(yv.shape).cpu().numpy()

        self.ax_dyn.quiver(xv, yv, grad_x, grad_y)
        self.ax_dyn.set_xlabel(r'$x$')
        self.ax_dyn.set_ylabel(r'$\dot{x}$')
        self.ax_dyn.legend(handles=self.legend_traj, loc='upper center', ncol=2)

    def save(self, dir, epoch):
         # IH: uncomment this when vis dyn
         #self.fig_dyn.savefig('{}dyn_epoch{}.png'.format(dir, epoch))
         self.fig_traj_split.savefig('{}viz_genes_epoch{}.png'.format(dir, epoch))