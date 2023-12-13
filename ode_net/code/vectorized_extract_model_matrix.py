# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
from odenet import ODENet

import os
import subprocess

base_dir = "C:/Users/Intekhab Hossain/Desktop/rebuttal/sim_690/" 
r_script_path = "C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/markdown_items/grn_ootb_to_be_called_by_vec_extract_mod.R"

# Loop over noise levels
for noise_level_folder in os.listdir(base_dir):
    print("\n\n")
    noise_level_path = os.path.join(base_dir, noise_level_folder)

    # Check if the item is a directory
    if os.path.isdir(noise_level_path):
        print(f"Noise Level: {noise_level_folder}")
        if noise_level_folder not in ['noise_0.025_experiments_for_rebuttal', 'noise_0.05_experiments_for_rebuttal']:

            # Loop over subfolders
            for subfolder in os.listdir(noise_level_path):
                if subfolder in ['tanh']:
                    subfolder_path = os.path.join(noise_level_path, subfolder)

                    # Check if the item is a directory
                    if os.path.isdir(subfolder_path):
                        print(f"\n  Subfolder: {subfolder}")
                        rep = 0
                        all_tanh_reps = os.listdir(subfolder_path)
                        for subsubfolder in [all_tanh_reps[0]]:
                            rep += 1
                            subsubfolder_path = os.path.join(noise_level_path, subfolder, subsubfolder)
                            subsubfolder_path = subsubfolder_path.replace("\\","/")
                            print(f"    Rep: {rep}")

                            ootb_model = torch.load('{}/best_val_model_ootb.pt'.format(subsubfolder_path))

                            Wo_1 = np.transpose(ootb_model.linear_1.weight.detach().numpy())
                            Bo_1 = np.transpose(ootb_model.linear_1.bias.detach().numpy())
                            Wo_2 = np.transpose(ootb_model.linear_out.weight.detach().numpy())
                            Bo_2 = np.transpose(ootb_model.linear_out.bias.detach().numpy())

                            np.savetxt("{}/wo_1.csv".format(subsubfolder_path), Wo_1, delimiter=",")
                            np.savetxt("{}/wo_2.csv".format(subsubfolder_path), Wo_2, delimiter=",")
                            np.savetxt("{}/bo_1.csv".format(subsubfolder_path), Bo_1, delimiter=",")
                            np.savetxt("{}/bo_2.csv".format(subsubfolder_path), Bo_2, delimiter=",")

                            '''
                            sums_model = torch.load('{}/best_val_model_sums.pt'.format(subsubfolder_path))
                            prods_model = torch.load('{}/best_val_model_prods.pt'.format(subsubfolder_path))
                            alpha_comb = torch.load('{}/best_val_model_alpha_comb.pt'.format(subsubfolder_path))
                            gene_mult = torch.load('{}/best_val_model_gene_multipliers.pt'.format(subsubfolder_path))
                            
                            Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
                            Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
                            Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
                            Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
                            alpha_comb = np.transpose(alpha_comb.linear_out.weight.detach().numpy())
                            gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())


                            num_features = alpha_comb.shape[0]
                            effects_mat = np.matmul(Wo_sums,alpha_comb[0:num_features//2]) + np.matmul(Wo_prods,alpha_comb[num_features//2:num_features])

                            num_cols = effects_mat.shape[1]
                            effects_mat = effects_mat * np.transpose(gene_mult)
                            effects_mat_path = "{}/effects_mat.csv".format(subsubfolder_path)
                            np.savetxt(effects_mat_path, effects_mat, delimiter=",")

                            
                            '''
                            

                            command = f"Rscript {r_script_path} {subsubfolder_path}"
                            subprocess.run(command, shell=True)
