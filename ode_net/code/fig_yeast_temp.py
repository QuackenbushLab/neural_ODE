import sys
import os
import numpy as np
import csv 
import math 

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

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
    
    neuron_dict = {"sim350": 40, "sim690": 50, 'yeast':80}
    models = ["phoenix_noprior", "phoenix"]
    datasets = ["yeast"]
    #noises = [0,0.025, 0.05, 0.1]
    
    dir_yeast = 'C:/STUDIES/RESEARCH/neural_ODE/pramila_yeast_data/clean_data/pramila_3551genes_2samples_24T.csv'
    
    data_handler_yeast = DataHandler.fromcsv(dir_yeast, "cpu", 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)

    datahandler_dict = {"yeast": data_handler_yeast}
    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_noprior" :"Unregularized PHOENIX (no prior)"} 
    gene_to_plot_dict = {"yeast": [4, 136, 1000, 3000, 200]}
    colors = ['orange','red','blue','green', 'pink', 'brown']
    
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_yeast_res = plt.figure(figsize=(8,8)) # tight_layout=True
    axes_yeast_res = fig_yeast_res.subplots(nrows= len(models), ncols=len(datasets), 
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_yeast_res.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 15
    
    plt.grid(True)
    
    print("......")
    
    for this_data in datasets:
        this_data_handler = datahandler_dict[this_data]
        this_neurons = neuron_dict[this_data]
        genes = gene_to_plot_dict[this_data]
        
        this_odenet = ODENet("cpu", this_data_handler.dim, explicit_time=False, neurons = this_neurons)
        this_odenet.float()
        for this_model in models:
            print("Now on model = {}".format(this_model))
            
            pretrained_model_file = 'C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/{}/{}/best_val_model.pt'.format(this_data, this_model)
            this_odenet.load(pretrained_model_file)
             
            trajectories, all_plotted_samples, extrap_timepoints = this_data_handler.calculate_trajectory(this_odenet, 'dopri5', num_val_trajs = 1, fixed_traj_idx = [1])
            times = this_data_handler.time_np
            data_np_to_plot = [this_data_handler.data_np[i] for i in all_plotted_samples]
            data_np_0noise_to_plot = [this_data_handler.data_np_0noise[i] for i in all_plotted_samples]

            col_num = 0
            row_num = models.index(this_model)
            ax = axes_yeast_res[row_num]
            ax.spines['bottom'].set_linewidth(border_width)
            ax.spines['left'].set_linewidth(border_width)
            ax.spines['top'].set_linewidth(border_width)
            ax.spines['right'].set_linewidth(border_width)
            ax.cla()

            ax.set_ylim((-1, 1))
            ax.set_xlim((0,300))
            for sample_idx, (approx_traj, traj, true_mean) in enumerate(zip(trajectories, data_np_to_plot, data_np_0noise_to_plot)):
                for gene,this_col in zip(genes, colors):
                    with torch.no_grad():
                        this_pred_traj = approx_traj[:,:,gene].numpy().flatten()
                        ax.plot(extrap_timepoints, this_pred_traj,
                            color = this_col, linestyle = "--", lw=2, label = "prediction") #times[sample_idx].flatten()[0:] 
                        
                        noisy_traj =  traj[:,:,gene].flatten()
                        observed_times = times[sample_idx].flatten()
                        ax.plot(observed_times, noisy_traj,    
                        color = this_col, lw = 5, linestyle = '-', alpha=0.3)
                        ax.plot(observed_times, noisy_traj, 
                        linestyle = 'None',
                        markerfacecolor = this_col, markeredgecolor = 'black', marker = "o",  alpha=0.8, markersize=7, label = gene)
        
            
            ax.tick_params(axis='x', labelsize= tick_lab_size)
            ax.tick_params(axis='y', labelsize= tick_lab_size)
                
            if col_num == 0:
                ax.set_ylabel(model_labels[this_model], fontsize=ax_lab_size)
            if row_num == 0:
                ax.set_title("Yeast data", fontsize = ax_lab_size) 
                
   #  cbar.set_ticks([0, 0.35, -0.35])
   # cbar.set_ticklabels(['None', 'Activating', 'Repressive'])
   # cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    #cbar.set_label('Estimated effect of '+ r'$g_i$'+ ' on ' +r"$\frac{dg_j}{dt}$" +' (based on trained model)', size = ax_lab_size)
    #cbar.outline.set_linewidth(2)

    
    fig_yeast_res.savefig('{}/manuscript_fig_yeast_res.png'.format(output_root_dir), bbox_inches='tight')    