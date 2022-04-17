import os
import numpy as np
import csv 

from visualization_inte import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np

if __name__ == "__main__":

    save_file_name = "just_plots"
    output_root_dir = '{}/{}/'.format('output', save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    num_gene_dict = {"sim350": 350, "sim690": 690}
    models = ["ootb_sigmoid", "ootb_relu","ootb_tanh", "phoenix_noprior", "phoenix" ]
    datasets = ["sim350", "sim690"]
    noises = [0, 0.1]
    perf_info = {}
    metrics = ['true_val_MSE']
    metric_labels = {'true_val_MSE':'MSE'}
    model_colors = {"phoenix":"green", "phoenix_noprior" :"red", 
                    "ootb_tanh" : "purple", "ootb_relu" : "purple", "ootb_sigmoid" : "purple"} 
    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_noprior" :"Unregularized PHOENIX",
                    "ootb_tanh" : "OOTB NeuralODE (tanh)",
                    "ootb_relu" : "OOTB NeuralODE (ReLU)",
                    "ootb_sigmoid" : "OOTB NeuralODE (sigmoid)"} 

    model_hatch = {"phoenix":"", "phoenix_noprior" :"", 
                    "ootb_tanh" : "///", "ootb_relu" : "..", "ootb_sigmoid" : "---"}                
    
    leg_general_info = [Patch(facecolor=model_colors[this_model], edgecolor= "black", 
                            alpha = 0.5, hatch = model_hatch[this_model],
                         label= model_labels[this_model]) for this_model in models]
                     
    for this_data in datasets:
        if this_data not in perf_info:
            perf_info[this_data] = {}
        for this_model in models:
            if this_model not in perf_info[this_data]:
                perf_info[this_data][this_model] = {}
            for this_noise in noises: 
                if this_noise not in perf_info[this_data][this_model]:
                    perf_info[this_data][this_model][this_noise] = {this_metric: 0 for this_metric in metrics}
    
    perf_csv = csv.DictReader(open('C:/STUDIES/RESEARCH/neural_ODE/all_manuscript_models/perf_plotting.csv', 'r'))
    for line in perf_csv: 
        if line['model'] in models and float(line['noise']) in noises:
            for this_metric in metrics:
                perf_info[line['dataset'].lower()][line['model']][float(line['noise'])][this_metric] = float(line[this_metric])

    #Plotting setup
    fig_ootb_mse = plt.figure(figsize=(6,8))
    #plt.grid(True)
    axes_ootb_mse = fig_ootb_mse.subplots(nrows= len(datasets),ncols = 1,  
    sharex=False, sharey=True, 
    subplot_kw={'frameon':True})
    #fig_ootb_mse.subplots_adjust(hspace=0.0, wspace=0.0)
    border_width = 1.5
    tick_lab_size = 11
    ind = np.arange(len(noises))  # the x locations for the groups
    width = 0.15  # the width of the bars
    deltas = [-2, -1, 0, 1, 2]
    print("......")
    
    for this_data in datasets:
        print("Now on data = {}".format(this_data))
        for this_model in models:
            #print("Now on data = {}, noise = {}".format(this_data, this_noise))
            
            row_num = datasets.index(this_data)
            ax = axes_ootb_mse[row_num]
            ax.spines['bottom'].set_linewidth(border_width)
            ax.spines['left'].set_linewidth(border_width)
            ax.spines['top'].set_linewidth(border_width)
            ax.spines['right'].set_linewidth(border_width)
            ax.tick_params(axis='x', labelsize= tick_lab_size)
            ax.tick_params(axis='y', labelsize= tick_lab_size)
            ax.set_ylim((6*10**-4,5*10**-3))
            ax.set_xticks(ind)
            ax.set_xticklabels(['No noise', 'High noise'])
            
            this_delta = deltas[models.index(this_model)] 
            this_model_mses =  [perf_info[this_data][this_model][this_noise]['true_val_MSE'] for this_noise in noises]
            ax.bar(ind + width*this_delta, this_model_mses, width = width, 
                    color = model_colors[this_model], alpha = 0.5,  edgecolor = "black", 
                    linewidth = 1.5, align = 'center', hatch = model_hatch[this_model])
            #ax.cla()
        ax.set_ylabel('validation MSE ({})'.format(this_data.upper()), fontsize=15)
        if row_num == 1:
            ax.set_xlabel('Noise level', fontsize=15)
            #if row_num == 0:
            #    this_title = "Noise level = {}".format(this_noise) 
            #    ax.set_title(this_title, fontsize = ax_lab_size, pad = 15)    
    plt.yscale("log")
    fig_ootb_mse.legend(handles = leg_general_info, loc='upper center', prop={'size': 11}, 
                        ncol = 2,  handleheight=1.5)
    fig_ootb_mse.savefig('{}/manuscript_fig_ootb_mse.png'.format(output_root_dir), bbox_inches='tight')
    