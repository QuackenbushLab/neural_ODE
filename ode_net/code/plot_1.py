import sys
import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from solve_eq import solve_eq
from visualization_inte import *

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.gridspec import SubplotSpec
#from figure_saver import save_figure
import numpy as np
import torch
import random 

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format('val_config_inte.cfg'))
    settings = read_arguments_from_file('val_config_inte.cfg')
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    dir_350 = 'C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/chalmers_350genes_150samples_earlyT_0bimod_1initvar.csv'
    dir_690 = 'C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/chalmers_690genes_150samples_earlyT_0bimod_1initvar.csv'

    data_handler_350 = DataHandler.fromcsv(dir_350, "cpu", 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)

    data_handler_690 = DataHandler.fromcsv(dir_690, "cpu", 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)
    
    datahandler_dict = {"sim350": data_handler_350, "sim690": data_handler_690}
    neuron_dict = {"sim350": 40, "sim690": 50}
    models = ["phoenix"]
    datasets = ["sim350", "sim690"]
    noises = [0, 0.025, 0.05, 0.1]
    
    #Plotting setup
    fig_traj_split = plt.figure(figsize=(15,15)) # tight_layout=True
    fig_traj_split.canvas.set_window_title("Trajectories in each dimension")
    axes_traj_split = fig_traj_split.subplots(nrows= len(datasets), ncols=len(noises), sharex=True, sharey=True, subplot_kw={'frameon':True})
    fig_traj_split.subplots_adjust(hspace=0, wspace=0)
    gene_to_plot_dict = {"sim350": [4, 136, 200, 275], "sim690": [20, 100, 275, 320]} #100
    colors = ['orange','red','blue','green', 'pink', 'brown']
    plt.grid(True)
    

    for this_data in datasets:
        this_data_handler = datahandler_dict[this_data]
        this_neurons = neuron_dict[this_data]
        genes = gene_to_plot_dict[this_data]
        for this_model in ["phoenix"]:
            this_odenet = ODENet("cpu", this_data_handler.dim, explicit_time=False, neurons = this_neurons)
            this_odenet.float()
                
            for this_noise in noises:
                print("Now on data = {}, noise = {}".format(this_data, this_noise))
                noise_string = "noise_{}".format(this_noise)
                pretrained_model_file = 'C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/{}/{}/{}/best_val_model.pt'.format(this_data, this_model, noise_string)
                this_odenet.load(pretrained_model_file)

                trajectories, all_plotted_samples, extrap_timepoints = this_data_handler.calculate_trajectory(this_odenet, 'dopri5', num_val_trajs = 1, fixed_traj_idx = [1])

                times = this_data_handler.time_np
                data_np_to_plot = [this_data_handler.data_np[i] for i in all_plotted_samples]
                data_np_0noise_to_plot = [this_data_handler.data_np_0noise[i] for i in all_plotted_samples]
                
                row_num = datasets.index(this_data)
                this_row_plots = axes_traj_split[row_num]
                col_num = noises.index(this_noise)
                ax = this_row_plots[col_num]
                ax.cla()
                ax.set_xlim((-0.5, 10.5))
                ax.set_ylim((-0.1,1.1))
                for sample_idx, (approx_traj, traj, true_mean) in enumerate(zip(trajectories, data_np_to_plot, data_np_0noise_to_plot)):
                    for gene,this_col in zip(genes, colors):
                        with torch.no_grad():
                            ax.plot(times[sample_idx].flatten()[0:], approx_traj[:,:,gene].numpy().flatten(),
                             color = this_col, linestyle = "dashdot", lw=1, label = "prediction") # self.extrap_timepoints
                            ax.plot(times[sample_idx].flatten(), approx_traj[:,:,gene].numpy().flatten() + np.random.normal(0,this_noise,5), markerfacecolor = this_col, markeredgecolor = 'black', marker = "o", linestyle = 'None', alpha=0.5, label = gene)
                            #ax.plot(times[sample_idx].flatten(), traj[:,:,gene].flatten() + np.random.normal(0,this_noise,5), markerfacecolor = this_col, markeredgecolor = 'black', marker = "o", linestyle = 'None', alpha=0.5, label = gene)
                            
                            #ax.plot(times[sample_idx].flatten(), true_mean[:,:,gene].flatten(),'g-', lw=1.5, alpha = 0.5) #
            
                ax.set_xlabel(r'$t$')
                #ax.legend()
    
    handles, labels = ax.get_legend_handles_labels()
    fig_traj_split.legend(handles, labels, loc='upper center', ncol = 4)
    fig_traj_split.savefig('{}/manuscript_fig_2.png'.format(output_root_dir))

    


    