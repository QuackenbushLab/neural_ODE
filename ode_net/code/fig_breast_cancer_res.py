import sys
import os
import numpy as np
import csv 

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
    fig_breast_cancer = plt.figure(figsize=(13,25), tight_layout=True)
    gs = fig_breast_cancer.add_gridspec(11, 6, hspace = 0.03, wspace = 0.02) #
    border_width = 1.5
    tick_lab_size = 14
    ax_lab_size = 15
    all_genes = [500, 2000, 4000, 11165]
    
    print("......")

    this_data = "breast_cancer"

    ax = fig_breast_cancer.add_subplot(gs[1:11, 0:5])
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.cla()
    
    print("making heatmap")
    z = np.loadtxt(open("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_inferred_influences_wide_FULL.csv", "rb"), 
        dtype = "str",delimiter=",", skiprows=1, usecols = (1,2,3,4))
    num_tops = z.shape[0]
    print("The analysis contains", num_tops, "genes.")
    num_models = z.shape[1]
    z[z == ""] = "0"                
    z = z.astype(float)
    z[ z == 0] = np.nanmin(z)#np.nanmin(z)
    z = z.transpose()
    
    ind = np.arange(num_models) +0.5
    ax.set_xlim(0,num_models)
    ax.set_ylim(0,num_tops)

    gene_names = np.loadtxt(open("/home/ubuntu/neural_ODE/all_manuscript_models/breast_cancer/all_inferred_influences_wide_FULL.csv", "rb"), 
        dtype = "str",delimiter=",", skiprows=1, usecols = (0))
    gene_names = np.char.strip(gene_names, '"')

    z_min, z_max = np.nanmin(z), np.nanmax(z)
    c = ax.pcolormesh(z.transpose(), cmap='Blues',# vmin=z_min, vmax= z_max, #120
            norm = colors.PowerNorm(gamma = 1.3)) #, #gamma = 2 shading = "nearest"
    
    for idx in range(num_tops):
        ax.axhline(y=idx, xmin=0, xmax=num_models, linestyle='dotted', color = "black", alpha = 0.3)

    for idx in range(num_models):
        ax.axvline(x=idx, ymin=0, ymax=num_tops, linestyle='dotted', color = "black", alpha = 0.3)    

    # for i in range(num_tops):
    #     for j in range(num_models):
    #         if z[j,i] <= 5:
    #             text = ax.text(j + 0.5, i + 0.5, int(z[j, i]),
    #                         ha="center", va="center", color="w",
    #                         size = 13, weight = "bold")

    
   
    '''
    print("......")
    print("overlaying performance metrics")
    ax1 = fig_breast_cancer.add_subplot(gs[0, 0:5], sharex = ax)
    #ax1.spines['bottom'].set_linewidth(border_width)
    #ax1.spines['left'].set_linewidth(border_width)
    #ax1.spines['top'].set_linewidth(border_width)
    #ax1.spines['right'].set_linewidth(border_width)
    ax1.set_frame_on(False)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.cla()
    
    perf_info = {}
    metrics = ['var_explained', 'AUC','runtime_cost' ]
    metric_cols = {'var_explained': 'green', 'AUC': 'orange','runtime_cost':'purple', 'true_harm_cent': "plum"}
    metric_labels = {'var_explained': r'$R^2$', 'AUC': 'AUC','runtime_cost':'AWS ($)', 'true_harm_cent': r"$\log(\mathcal{HC}_{ChIPSeq})$" } 
    metrics_extended = ['var_explained', 'AUC','runtime_cost', 'true_harm_cent' ]

    leg_general_info = [Patch(facecolor=metric_cols[this_metric], edgecolor= "black", linewidth = 1.5,
                            alpha = 0.7, label= metric_labels[this_metric]) for this_metric in metrics_extended]

    for this_gene in all_genes:
        if this_gene not in perf_info:
            perf_info[this_gene] = {this_metric: 0 for this_metric in metrics}

    perf_csv = csv.DictReader(open('C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/bc_model_perfs.csv', 'r'))
    for line in perf_csv: 
        for this_metric in metrics:
            perf_info[float(line['num_gene'])][this_metric] = float(line[this_metric])

    print(perf_info)

    width= 0.17  # the width of the bars
    deltas = [-1, 0, 1]
    cost_shrinker = 1.8
    
    for this_metric in metrics:
        this_delta = deltas[metrics.index(this_metric)] 
        this_perf_vals = [perf_info[this_gene][this_metric]/cost_shrinker if this_metric == "runtime_cost" else perf_info[this_gene][this_metric] for this_gene in all_genes ]
        ax1.bar(ind  + width*this_delta, this_perf_vals, width = width, 
                        capsize = 5, color = metric_cols[this_metric], alpha = 0.7,  edgecolor = "black", 
                        linewidth = 1.5, align = 'center')
        
    for this_metric in metrics:
        this_delta = deltas[metrics.index(this_metric)] 
        this_perf_vals = [perf_info[this_gene][this_metric] for this_gene in all_genes]
        for this_idx in range(len(this_perf_vals)):
            this_val = this_perf_vals[this_idx]
            this_text = this_val
            this_height = this_val
            if this_metric == "runtime_cost":
                this_text = "${:,.2f}".format(this_val)
                this_height = this_val/cost_shrinker
            ax1.text(this_idx + 0.5 + width*this_delta, this_height + 0.1 , this_text, 
                     ha="center", va="bottom", color="black",size = 15, rotation = 90)
    
    ax1.legend(handles = leg_general_info, loc='upper right', prop={'size': tick_lab_size+1}, 
                        ncol = 1,  handleheight=1.5, frameon = False,  bbox_to_anchor = (1.25,1.2)) #
    
    print("......")
    print("overlaying true centralities")
    ax2 = fig_breast_cancer.add_subplot(gs[1:11, 5], sharey = ax)     
    ax2.set_frame_on(True)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    ax2.cla()

    true_vals = np.loadtxt(open("C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/breast_cancer/all_inferred_influences_wide.csv", "rb"), 
        dtype = "str",delimiter=",", skiprows=1, usecols = (num_models +1))
    true_vals[true_vals == ""] = "0"
    #true_vals[true_vals == "0"] = "1"
    true_vals = true_vals.astype(float)

    ax2.barh(np.arange(num_tops)+0.5, true_vals, height = 1, 
                       color = metric_cols['true_harm_cent'], alpha = 0.7, align = 'center', 
                       edgecolor = "black", linewidth = 1.5)

    for this_top in range(num_tops):
        this_val = true_vals[this_top]
        if this_val >0: 
            ax2.text(this_val + 100, this_top + 0.5, "{:.2f}".format(np.log(this_val)), 
            ha="left", va="center", color="black",size = 15, rotation = 0)
    ax2.set_xscale("log")
    
    #ax2.xaxis.tick_top()
    
    #ax2.grid(visible = True, which = "both", axis = "x", color = "black", 
    #        linestyle = "--", alpha = 0.3)
    
    '''
    ax.set_yticks(np.arange(num_tops)+0.5)    
    ax.set_yticklabels(gene_names)
    ax.tick_params(axis='y', labelsize= tick_lab_size+1)
    
    ax.set_xticks(ind)    
    ax.set_xticklabels([ r"$n_g=$ "+str(this_tot_gene) for this_tot_gene in all_genes])
    ax.tick_params(axis='x', labelsize= tick_lab_size + 5)
    

    cbar =  fig_breast_cancer.colorbar(c, ax= [ax], use_gridspec = False,  #[ax, ax2]
                            shrink=0.6, orientation = "horizontal", pad = 0.03)
    
    #cbar.set_ticks([0 , 10, 15, 20])
    cbar.ax.tick_params(labelsize = tick_lab_size) 
    cbar.set_label("Inferred influence score " + r'$(\mathcal{IS}_{inferred})$', size = ax_lab_size+3)
    cbar.outline.set_linewidth(2)
    #plt.subplots_adjust(wspace=0, hspace=0)

    fig_breast_cancer.savefig('{}/manuscript_fig_breast_cancer_influence_rebuttal.png'.format(output_root_dir), bbox_inches='tight')
