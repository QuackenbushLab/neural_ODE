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
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format('output', save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
                    
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_heat_native = plt.figure(figsize=(10,18)) # tight_layout=True
    axes_heat_sparse = fig_heat_native.subplots(ncols= 1, nrows=1, 
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_heat_native.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 15
    
    plt.grid(True)
    
    print("......")
    
        #this_data_handler = datahandler_dict[this_data]
    ax = axes_heat_sparse
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.cla()

    num_genes = 350 
    
    y, x = np.meshgrid(np.linspace(1, num_genes, num_genes), np.linspace(1, num_genes, num_genes))
    native_dir = "C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/edge_prior_matrix_chalmers_350.csv"
    z = np.loadtxt(open(native_dir, "rb"), delimiter=",", skiprows=1)

    color_mult = 1 #0.25
    z_min, z_max = color_mult*-np.abs(z).max(), color_mult*np.abs(z).max()
    c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max) 
    ax.axis([x.min(), x.max(), y.min(), y.max()]) 
    
    fig_heat_native.canvas.draw()
    labels_y = [item.get_text() for item in ax.get_yticklabels()]
    labels_y_mod = [(r"$g'$"+item).translate(SUB) for item in labels_y]
    labels_x = [item.get_text() for item in ax.get_xticklabels()]
    labels_x_mod = [(r'$g$'+item).translate(SUB) for item in labels_x]
    
    ax.set_xticklabels(labels_x_mod)
    ax.set_yticklabels(labels_y_mod)
    ax.tick_params(axis='x', labelsize= tick_lab_size)
    ax.tick_params(axis='y', labelsize= tick_lab_size)
        
    ax.set_title("Native matrix", fontsize=ax_lab_size, pad = 10)
                
    cbar =  fig_heat_native.colorbar(c, ax=axes_heat_sparse, 
                                        shrink=0.95, orientation = "horizontal", pad = 0.05)
    #cbar.set_ticks([0, 0.03, -0.03])
    #cbar.set_ticklabels(['None', 'Activating', 'Repressive'])
    cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    cbar.set_label(r'$\widetilde{D_{ij}}$= '+'Estimated effect of '+ r'$g_j$'+ ' on ' +r"$\frac{dg_i}{dt}$" +' in SIM350', size = ax_lab_size)
    cbar.outline.set_linewidth(2)

    
    fig_heat_native.savefig('{}/manuscript_fig_heat_native.png'.format(output_root_dir), bbox_inches='tight')
    