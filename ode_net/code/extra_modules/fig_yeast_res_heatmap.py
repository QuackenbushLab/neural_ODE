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
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_yeast_res = plt.figure(figsize=(15,10)) # tight_layout=True
    axes_yeast_res = fig_yeast_res.subplots(ncols= len(models), nrows=len(datasets), 
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
        this_odenet = ODENet("cpu", this_data_handler.dim, explicit_time=False, neurons = this_neurons)
        this_odenet.float()
        for this_model in models:
            print("Now on model = {}".format(this_model))
            
            row_num = 0
            col_num = models.index(this_model)
            ax = axes_yeast_res[col_num]
            ax.spines['bottom'].set_linewidth(border_width)
            ax.spines['left'].set_linewidth(border_width)
            ax.spines['top'].set_linewidth(border_width)
            ax.spines['right'].set_linewidth(border_width)
            ax.cla()

            pretrained_model_file = 'C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/{}/{}/best_val_model.pt'.format(this_data, this_model)
            this_odenet.load(pretrained_model_file)
            Wo_sums = np.transpose(this_odenet.net_sums.linear_out.weight.detach().numpy())
            Wo_prods = np.transpose(this_odenet.net_prods.linear_out.weight.detach().numpy())
            alpha_comb = np.transpose(this_odenet.net_alpha_combine.linear_out.weight.detach().numpy())
            gene_mult = np.transpose(torch.relu(this_odenet.gene_multipliers.detach()).numpy()) 
            
            num_gene = 3551
            y, x = np.meshgrid(np.linspace(1, num_gene, num_gene), np.linspace(1, num_gene, num_gene))
            z = np.matmul(Wo_sums, alpha_comb[0:this_neurons,]) + np.matmul(Wo_prods, alpha_comb[this_neurons:(2*this_neurons),])    
            z = z* gene_mult.reshape(1, -1)
            if this_model == "phoenix_noprior":
                z = z*4
            #row_sums =  z.sum(axis=1)
            #z = np.transpose(z / row_sums[:, np.newaxis])

            #color_mult =0.1
            z_min, z_max = -np.abs(z).max(), np.abs(z).max()
            c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin= -0.2, vmax= 0.2) #  
            ax.axis([x.min(), x.max(), y.min(), y.max()]) 
            fig_yeast_res.colorbar(c, ax =ax, shrink=0.95, orientation = "horizontal", pad = 0.05)

            if row_num == 0 and col_num == 0:
                fig_yeast_res.canvas.draw()
                labels_x = [item.get_text() for item in ax.get_xticklabels()]
                labels_x_mod = [(r"$g'$"+item).translate(SUB) for item in labels_x]
                labels_y = [item.get_text() for item in ax.get_yticklabels()]
                labels_y_mod = [(r'$g$'+item).translate(SUB) for item in labels_y]
            
            ax.set_xticklabels(labels_x_mod)
            ax.set_yticklabels(labels_y_mod)
            ax.tick_params(axis='x', labelsize= tick_lab_size)
            ax.tick_params(axis='y', labelsize= tick_lab_size)
                
            if row_num == 0:
                ax.set_title(model_labels[this_model], fontsize=ax_lab_size, pad = 10)
            if col_num == 0:
                ax.set_ylabel("Yeast data", fontsize = ax_lab_size) 
                
   #  cbar.set_ticks([0, 0.35, -0.35])
   # cbar.set_ticklabels(['None', 'Activating', 'Repressive'])
   # cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    #cbar.set_label('Estimated effect of '+ r'$g_i$'+ ' on ' +r"$\frac{dg_j}{dt}$" +' (based on trained model)', size = ax_lab_size)
    #cbar.outline.set_linewidth(2)

    
    fig_yeast_res.savefig('{}/manuscript_fig_yeast_heatmap.png'.format(output_root_dir), bbox_inches='tight')    