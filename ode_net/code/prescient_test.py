# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np

import torch

from helper_for_prescient import *
from visualization_inte import *



def _build_save_file_name(save_path, epochs):
    return 'prescient_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))



# Main function
if __name__ == "__main__":
    
    create_data_only = False
    create_velo_only = False
    
    if create_data_only :
        prescient_data_dict = prescient_read_data(data_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_input_350.csv",
            meta_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_meta_350.csv",
            out_dir = "/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_350_",
            tp_col = "timepoint", celltype_col = "cell_type")

        print("saving expression data")
        torch.save(prescient_data_dict, prescient_data_dict["out_dir"]+"data.pt")
        quit()
    
    if create_velo_only:
        prescient_data_dict = prescient_read_data(data_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_true_vel_350.csv",
            meta_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_meta_350.csv",
            out_dir = "/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_350_",
            tp_col = "timepoint", celltype_col = "cell_type")

        print("saving true velo data")
        torch.save(prescient_data_dict, prescient_data_dict["out_dir"]+"true_vel.pt")
        quit()
        #python -m prescient train_model -i/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_350_data.pt --weight_name "regular" --train_epochs 2500 --save 500

    print("......................")

    #Q does normalization improve things?
    #Q how to incorporate noise into all of this
    #Q do more principled train-val split

    print("now get the fitted PRESCIENT model")
    device = torch.device('cpu')
    seed = 2
    model_path = "/home/ubuntu/neural_ODE/ode_net/code/experiments/regular-softplus_1_500-1e-06/"
    # load model
    config_path = os.path.join(str(model_path), 'seed_{}/config.pt'.format(seed))
    config = SimpleNamespace(**torch.load(config_path))
    net = AutoGenerator(config)

    train_pt = os.path.join(model_path, 'seed_{}/train.best.pt'.format(seed))
    checkpoint = torch.load(train_pt, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)    

    data_reload = torch.load("/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_350_data.pt")
    true_velos_load = torch.load("/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_350_true_vel.pt")
    X_full = torch.stack(data_reload['x'])
    X_full = X_full + torch.randn(X_full.shape)
    true_velos_full = torch.stack(true_velos_load['x'])
    pred_velos_full = net._drift(X_full)
    corr_coeff_full = round(np.corrcoef(true_velos_full.detach().flatten(), 
                                        pred_velos_full.detach().flatten())[0,1], 4)
    print(corr_coeff_full)
    print("......................")
    
