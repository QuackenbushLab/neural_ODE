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
import matplotlib.colors as colors
import numpy as np
import torch

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format("output", save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    #models = ["phoenix"]
    datasets = ["breast_cancer"]
    
    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_breast_cancer = plt.figure(figsize=(10,18)) # tight_layout=True
    axes_breast_cancer = fig_breast_cancer.subplots(ncols= 1, nrows=1, 
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_breast_cancer.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 15
    
    z = np.loadtxt(open("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/all_harm_cent_wide.csv", "rb"), 
                    dtype = "str",delimiter=",", skiprows=1, usecols = (1,2,3,4))

    gene_names = np.loadtxt(open("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/all_harm_cent_wide.csv", "rb"), 
                    dtype = "str",delimiter=",", skiprows=1, usecols = (0))

    gene_names = np.char.strip(gene_names, '"')

    num_tops = z.shape[0]
    print("The analysis contains", num_tops, "genes.")
    num_models = z.shape[1]
    z[z == ""] = "0"                
    z = z.astype(float)
    z[ z == 0] = np.nanmax(z)
    z = z.transpose()
    y, x = np.meshgrid(np.linspace(0, num_tops, num_tops), np.linspace(0, num_models+1, num_models+1))
                
    plt.grid(True)
    
    print("......")
    
    for this_data in datasets:
        #this_data_handler = datahandler_dict[this_data]
                
        ax = axes_breast_cancer#[row_num]
        ax.spines['bottom'].set_linewidth(border_width)
        ax.spines['left'].set_linewidth(border_width)
        ax.spines['top'].set_linewidth(border_width)
        ax.spines['right'].set_linewidth(border_width)
        ax.cla()

        
        
        z_min, z_max = np.nanmin(z), np.nanmax(z)
        c = ax.pcolormesh(z.transpose(), cmap='Blues_r', vmin=z_min, vmax=120, 
                            norm = colors.PowerNorm(gamma = 0.4),
                            shading = "nearest") 
        #ax.axis([x.min(), x.max(), y.min(), y.max()]) 
        
        fig_breast_cancer.canvas.draw()
        #labels_y = [item.get_text() for item in ax.get_yticklabels()]
        #labels_x = [item.get_text() for item in ax.get_xticklabels()]
        
        ax.set_yticks(np.arange(num_tops)+0.5)    
        ax.set_yticklabels(gene_names)
        ax.tick_params(axis='x', labelsize= tick_lab_size)
        ax.tick_params(axis='y', labelsize= tick_lab_size)
        
        for idx in range(num_tops):
            plt.axhline(y=idx, xmin=0, xmax=4, linestyle='dotted', color = "gray", alpha = 0.3)

        for i in range(num_tops):
            for j in range(num_models):
                if z[j,i] <= 5:
                    text = ax.text(j + 0.5, i + 0.5, int(z[j, i]),
                                ha="center", va="center", color="w",
                                size = 13, weight = "bold")
                
        #if row_num == 0:
        #    ax.set_title(model_labels[this_model], fontsize=ax_lab_size, pad = 10)
        #if col_num == 0:
        #    ax.set_ylabel("Noise level = {:.0%}".format(this_noise/0.5), fontsize = ax_lab_size) 

                 
    cbar =  fig_breast_cancer.colorbar(c, ax=axes_breast_cancer, 
                                        shrink=0.5, orientation = "vertical", pad = 0.05,
                                        extend = "max")
    #cbar.set_ticks([0, 0.03, -0.03])
    #cbar.set_ticklabels(['None', 'Activating', 'Repressive'])
    #cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    #cbar.set_label(r'$\widetilde{D_{ij}}$= '+'Estimated effect of '+ r'$g_j$'+ ' on ' +r"$\frac{dg_i}{dt}$" +' in SIM350', size = ax_lab_size)
    cbar.outline.set_linewidth(2)

    
    fig_breast_cancer.savefig('{}/manuscript_fig_breast_cancer.png'.format(output_root_dir), bbox_inches='tight')
    