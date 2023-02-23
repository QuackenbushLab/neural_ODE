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
from helper_true_velo import *


def _build_save_file_name(save_path, epochs):
    return 'prescient_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

#torch.set_num_threads(36)

# Main function
if __name__ == "__main__":
    
    create_data_only = False
    my_noise_to_test = 0.025
    
    if create_data_only :

        val_split = 0.09
        num_samples = 150
        n_dyn_val = int(num_samples*val_split)
        dyn_val_set = np.random.choice(range(num_samples), n_dyn_val, replace=False)
        dyn_train_set = np.setdiff1d(range(num_samples), dyn_val_set)

        prescient_data_dict = prescient_read_data(data_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_input_690.csv",
            meta_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_meta_690.csv",
            out_dir = "/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/data/sim690/noise_{}/prescient_690_".format(my_noise_to_test),
            tp_col = "timepoint", celltype_col = "celltype", 
            val_set = dyn_val_set, train_set = dyn_train_set, noise_sd = my_noise_to_test)
        print("saving expression data")
        torch.save(prescient_data_dict, prescient_data_dict["out_dir"]+"data.pt")
    
        prescient_true_velo_dict = prescient_read_data(data_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_true_velo_690.csv",
            meta_path = "/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/prescient_meta_690.csv",
            out_dir = "/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/data/sim690/noise_{}/prescient_690_".format(my_noise_to_test),
            tp_col = "timepoint", celltype_col = "celltype", 
            val_set = dyn_val_set, train_set = dyn_train_set)
        print("saving true velo data")
        torch.save(prescient_true_velo_dict, prescient_true_velo_dict["out_dir"]+"true_vel.pt")
        quit()
        #python -m prescient train_model -i /home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/prescient_690_data.pt --weight_name "regular_noise0" --train_epochs 500 --save 500 --out_dir /home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/experiments/

    print("......................")

    #Q does normalization improve things?
    #Q how to incorporate noise into all of this
    #Q do more principled train-val split

    print("now get the fitted PRESCIENT model")
    print("")
    
    device = torch.device('cpu')
    model_path = "/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/experiments/sim690/regular_noise{}-softplus_1_100-1e-06/seed_2".format(my_noise_to_test)
    # load model
    config_path = os.path.join(str(model_path), 'config.pt')
    config = SimpleNamespace(**torch.load(config_path))
    net = AutoGenerator(config)

    #for epoch_str in ['000100', '000200', '000300', '000400', '000500', '000600', '000700', '000800', '000900', '001000', '001100',  '001200']:
    for epoch_str in ['000200', '000400', '000600', '000800', '001000', '001200', '001400', '001600', '001800', '002000','002200','002400','002600']: 
        print("......................")
        print("Epoch:", epoch_str)
        train_pt = os.path.join(model_path, 'train.epoch_{}.pt'.format(epoch_str))
        #train_pt = os.path.join(model_path, 'train.best.pt'.format(epoch_str))
        checkpoint = torch.load(train_pt, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device)    

        data_reload = torch.load("/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/data/sim690/noise_{}/prescient_690_data.pt".format(my_noise_to_test))
        true_velos_load = torch.load("/home/ubuntu/neural_ODE/ode_net/code/output/_prescient_data/data/sim690/noise_{}/prescient_690_true_vel.pt".format(my_noise_to_test))
        
        X_train = torch.stack(data_reload['xp'])
        true_velos_train = torch.stack(true_velos_load['xp'])
        pred_velos_train = net._drift(X_train)
        corr_coeff_train = round(np.corrcoef(true_velos_train.detach().flatten(),
                                            pred_velos_train.detach().flatten())[0,1], 4)
        print("Training corr:",corr_coeff_train, "using", true_velos_train.shape[1], "samples")
        
        X_val = torch.stack(data_reload['x_val'])
        true_velos_val = torch.stack(true_velos_load['x_val'])
        pred_velos_val = net._drift(X_val)
        corr_coeff_val = round(np.corrcoef(true_velos_val.detach().flatten(),
                                            pred_velos_val.detach().flatten())[0,1], 4)
        print("Validation corr:",corr_coeff_val, "using", true_velos_val.shape[1], "samples")
        
        X_val_traj = torch.stack(data_reload['x_val_traj'])
        val_traj_shapes = X_val_traj.shape
        new_shape = (val_traj_shapes[0]*val_traj_shapes[1], val_traj_shapes[2])
        X_val_traj = torch.reshape(X_val_traj, new_shape)
        all_t_vals = data_reload['y']
        start_times = [ t  for t in all_t_vals[:-1] for i in range(val_traj_shapes[1])]
        end_times = [ t  for t in all_t_vals[1:] for i in range(val_traj_shapes[1])]
        t_val = np.array(list(zip(start_times, end_times)))

        velo_fun_x = lambda t,x : net._drift(torch.from_numpy(x).float()).detach().numpy()

        pred_next_pts = pred_traj_given_ode(my_ode_func = velo_fun_x, 
                                                    X_val = X_val_traj, 
                                                    t_val = t_val) #WORK ON THIS!

        X_val_traj_target = torch.stack(data_reload['x_val_traj_target'])
        X_val_traj_target = torch.reshape(X_val_traj_target, new_shape)
        mse_val_traj = torch.mean((X_val_traj_target - pred_next_pts)**2).item()
        print("MSE val traj = {:.3E}".format(mse_val_traj))   


    print("......................")
    train_pt = os.path.join(model_path, 'train.best.pt'.format(epoch_str))
    checkpoint = torch.load(train_pt, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)  

    X_train = torch.stack(data_reload['xp'])
    true_velos_train = torch.stack(true_velos_load['xp'])
    pred_velos_train = net._drift(X_train)
    corr_coeff_train = round(np.corrcoef(true_velos_train.detach().flatten(),
                                        pred_velos_train.detach().flatten())[0,1], 4)
    print("Best model training corr:",corr_coeff_train, "using", true_velos_train.shape[1], "samples")
        
    X_val = torch.stack(data_reload['x_val'])
    true_velos_val = torch.stack(true_velos_load['x_val'])
    pred_velos_val = net._drift(X_val)
    corr_coeff_val = round(np.corrcoef(true_velos_val.detach().flatten(),
                                        pred_velos_val.detach().flatten())[0,1], 4)
    print("Best model validation corr:",corr_coeff_val, "using", true_velos_val.shape[1], "samples")

    X_val_traj = torch.stack(data_reload['x_val_traj'])
    val_traj_shapes = X_val_traj.shape
    new_shape = (val_traj_shapes[0]*val_traj_shapes[1], val_traj_shapes[2])
    X_val_traj = torch.reshape(X_val_traj, new_shape)
    all_t_vals = data_reload['y']
    start_times = [ t  for t in all_t_vals[:-1] for i in range(val_traj_shapes[1])]
    end_times = [ t  for t in all_t_vals[1:] for i in range(val_traj_shapes[1])]
    t_val = np.array(list(zip(start_times, end_times)))

    velo_fun_x = lambda t,x : net._drift(torch.from_numpy(x).float()).detach().numpy()

    pred_next_pts = pred_traj_given_ode(my_ode_func = velo_fun_x, 
                                                X_val = X_val_traj, 
                                                t_val = t_val) #WORK ON THIS!

    X_val_traj_target = torch.stack(data_reload['x_val_traj_target'])
    X_val_traj_target = torch.reshape(X_val_traj_target, new_shape)
    mse_val_traj = torch.mean((X_val_traj_target - pred_next_pts)**2).item()
    print("MSE val traj = {:.3E}".format(mse_val_traj))   

    #velo traj is up here

        