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
import torch.optim as optim

import dynamo as dyn

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

def get_true_val_velocities(odenet, data_handler, method, batch_type):
    data_pw, t_pw, target_pw = data_handler.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    data_pw = data_pw + 0.025*torch.randn(size = data_pw.shape) 
    true_velo_pw, velo_t_pw, velo_target_pw = data_handler_velo.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    
    num_samples = data_pw.shape[0]
    n_dyn_val = int(num_samples*0.10)
    dyn_val_set = np.random.choice(range(num_samples), n_dyn_val, replace=False)
    dyn_train_set = np.setdiff1d(range(600), dyn_val_set)

    data_pw_train = data_pw[dyn_train_set, :, :]
    t_pw_train = t_pw[dyn_train_set, : ] #not used
    data_pw_val = data_pw[dyn_val_set, :, :]
    t_pw_val = t_pw[dyn_val_set, : ]

    true_velo_train = true_velo_pw[dyn_train_set, :, :]
    true_velo_val = true_velo_pw[dyn_val_set, :, :]

    phx_val_set_pred = odenet.forward(t_pw_val,data_pw_val)
    
    data_pw_train = torch.squeeze(data_pw_train).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw_val = torch.squeeze(data_pw_val).detach().numpy() #convert to NP array for dynamo VF alg
    true_velo_train = torch.squeeze(true_velo_train).detach().numpy() #convert to NP array for dynamo VF alg
    true_velo_val = torch.squeeze(true_velo_val).detach().numpy() #convert to NP array for dynamo VF alg
    phx_val_set_pred = torch.squeeze(phx_val_set_pred).detach().numpy() #convert to NP array for dynamo VF alg
 
    return {'x_train': data_pw_train, 'true_velo_x_train': true_velo_train, 
            'x_val': data_pw_val, 'true_velo_x_val': true_velo_val, 
            'phx_val_set_pred' : phx_val_set_pred} 




def _build_save_file_name(save_path, epochs):
    return 'dynamo_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name =  "chalmers_690genes_150samples_earlyT_0bimod_1initvar" 
clean_name_velo =  "chalmers_690genes_150samples_earlyT_0bimod_1initvar_DERIVATIVES" 
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name))
parser.add_argument('--velo_data', type=str, default='/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name_velo))


args = parser.parse_args()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    cleaned_file_name = clean_name
    save_file_name = _build_save_file_name(cleaned_file_name, settings['epochs'])

    if settings['debug']:
        print("********************IN DEBUG MODE!********************")
        save_file_name= '(DEBUG)' + save_file_name
    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)

    img_save_dir = '{}img/'.format(output_root_dir)
    interm_models_save_dir = '{}interm_models/'.format(output_root_dir)
    #intermediate_models_dir = '{}intermediate_models/'.format(output_root_dir)

    # Create image and model save directory
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(interm_models_save_dir):
        os.mkdir(interm_models_save_dir)

    # Save the settings for future reference
    with open('{}/settings.csv'.format(output_root_dir), 'w') as f:
        f.write("Setting,Value\n")
        for key in settings.keys():
            f.write("{},{}\n".format(key,settings[key]))

    # Use GPU if available
    if not settings['cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print("Trying to run on GPU -- cuda available: " + str(torch.cuda.is_available()))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on", device)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print("Running on CPU")
        device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = 0,
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])
    
    data_handler_velo = DataHandler.fromcsv(args.velo_data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = 0,
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])

    # Initialization
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                    log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
    odenet.float()
    param_count = sum(p.numel() for p in odenet.parameters() if p.requires_grad)
    param_ratio = round(param_count/ (data_handler.dim)**2, 3)
    print("Using a NN with {} neurons per layer, with {} trainable parameters, i.e. parametrization ratio = {}".format(settings['neurons_per_layer'], param_count, param_ratio))
    
    pretrained_model_file = '/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
    odenet.load(pretrained_model_file)
        
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
        net_file.write('\n\n\n')
        net_file.write(inspect.getsource(ODENet.forward))
        net_file.write('\n')

    #quit()

    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, odenet, settings)

    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    
    #DYNAMO vector field RKHS regression
    dynamo_vf_inputs = get_true_val_velocities(odenet, data_handler, settings['method'], settings['batch_type'])
    
    X_train = dynamo_vf_inputs['x_train']
    X_val = dynamo_vf_inputs['x_val']
    true_velos_train = dynamo_vf_inputs['true_velo_x_train']
    true_velos_val = dynamo_vf_inputs['true_velo_x_val']
    phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']

    print("..................................")
    print("PHX val corr vs true velos (w/o access!):", 
    round(np.corrcoef(true_velos_val.flatten(), phx_val_set_pred.flatten())[0,1], 4))
    print("..................................")
    
    best_val_corr = 0
    best_M = None
    best_vf = None

    for this_M in [30, 50, 100, 150, 200, 300, 400, 600]:

        print("Dynamo, M:", this_M)
        my_vf = dyn.vf.SvcVectorField(X = X_train, V = true_velos_train, 
                                        Grid = X_val,
                                        gamma = 1, M = this_M, lambda_ = 3) #gamma = 1 since we dont think there are any outliers
        trained_results = my_vf.train(normalize = False)
        dyn_pred_velos_train = trained_results['V']
        dyn_pred_velos_val = trained_results['grid_V']
        
        corr_coeff_train = round(np.corrcoef(true_velos_train.flatten(), 
                                        dyn_pred_velos_train.flatten())[0,1], 4)
        corr_coeff_val = round(np.corrcoef(true_velos_val.flatten(), 
                                        dyn_pred_velos_val.flatten())[0,1], 4)

        print("train corr:", corr_coeff_train, ", val_corr:", corr_coeff_val)
        if corr_coeff_val > best_val_corr:
            print("updating best VF!")
            best_val_corr = corr_coeff_val
            best_M = this_M
            best_vf = my_vf
        print("..................................")
        #print(trained_results['C'].shape)
    
    print("Best M:", best_M,"best val corr:", best_val_corr)
    print("obtaining Jacobian now..")

    #Jacobian analysis

    jac = best_vf.get_Jacobian()
    n_iter = 10000
    jac_sum = np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    for iter in range(n_iter -1):
        jac_sum = jac_sum + np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    
    jac_avg = np.transpose(jac_sum/n_iter)
    np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", jac_avg, delimiter=",")

    print("")
    print("saved abosulte average jacobian matrix, taken over", n_iter, "random points.")
    quit()
