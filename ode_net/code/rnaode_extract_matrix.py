# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
import sklearn
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
import torch.optim as optim


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

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import BaseDecisionTree

from GRN_rnaode import *

def BUILD_MODEL(counts, velocity, genes=None, tfs=None, method='rf', n_estimators=10, max_depth=10, lasso_alpha=1, train_size=0.7):
    '''
    v = f(x, a). return fitted function f
    method: 'rf'(random forest) / 'lasso' / 'linear'.
    '''
    if genes is None:
        genes = np.array([True] * counts.shape[1])
    if tfs is None:
        tfs = genes
    x, x_val, y, y_val = train_test_split(counts[:, tfs], velocity[:, genes], 
                                          test_size=1-train_size, random_state=42)
    
    # Build model
    if method == 'lasso':
        model = linear_model.Lasso(alpha=lasso_alpha)
    elif method == 'linear':
        model = linear_model.LinearRegression(n_jobs=-1)
    elif method == 'rf':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9, n_jobs=-1)  
    model = model.fit(x, y)    
    train_score = (model.score(x, y))**0.5
    test_score = (model.score(x_val, y_val))**0.5
    
    print('Fitted model | Training Corr: %.4f; Test Corr: %.4f' % (train_score, test_score))
    return model, train_score, test_score

def get_true_val_velocities(odenet, data_handler, method, batch_type):
    data_pw, unimportant_t_pw, _unused3 = data_handler.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    true_velo_pw, _unused1, _unused2 = data_handler_velo.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    
    noise = 0.0
    scale_factor_for_counts = 1
    val_split = 0.10

    data_pw = (data_pw+ noise*torch.randn(size = data_pw.shape))  * scale_factor_for_counts
    true_velo_pw = true_velo_pw * scale_factor_for_counts

    num_samples = data_pw.shape[0]
    n_dyn_val = int(num_samples*val_split)
    dyn_val_set = np.random.choice(range(num_samples), n_dyn_val, replace=False)
    dyn_train_set = np.setdiff1d(range(num_samples), dyn_val_set)

    data_pw_train = data_pw[dyn_train_set, :, :]
    data_pw_val = data_pw[dyn_val_set, :, :]
    t_pw_val = unimportant_t_pw[dyn_val_set, : ]

    true_velo_train = true_velo_pw[dyn_train_set, :, :]
    true_velo_val = true_velo_pw[dyn_val_set, :, :]

    phx_val_set_pred = scale_factor_for_counts* odenet.forward(t_pw_val,data_pw_val/scale_factor_for_counts)
    
    data_pw_train = torch.squeeze(data_pw_train).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw_val = torch.squeeze(data_pw_val).detach().numpy() #convert to NP array for dynamo VF alg
    true_velo_train = torch.squeeze(true_velo_train).detach().numpy() #convert to NP array for dynamo VF alg
    true_velo_val = torch.squeeze(true_velo_val).detach().numpy() #convert to NP array for dynamo VF alg
    phx_val_set_pred = torch.squeeze(phx_val_set_pred).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw = torch.squeeze(data_pw).detach().numpy()
    true_velo_pw = torch.squeeze(true_velo_pw).detach().numpy()

    return {'x_train': data_pw_train, 'true_velo_x_train': true_velo_train, 
            'x_val': data_pw_val, 'true_velo_x_val': true_velo_val, 
            'phx_val_set_pred' : phx_val_set_pred,
            'x_full': data_pw, 'true_velo_x_full': true_velo_pw} 




def _build_save_file_name(save_path, epochs):
    return 'rnaode_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name =  "chalmers_350genes_150samples_earlyT_0bimod_1initvar" 
clean_name_velo =  "chalmers_350genes_150samples_earlyT_0bimod_1initvar_DERIVATIVES" 
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
    
    #X_train = dynamo_vf_inputs['x_train']
    #X_val = dynamo_vf_inputs['x_val']
    #true_velos_train = dynamo_vf_inputs['true_velo_x_train']
    true_velos_val = dynamo_vf_inputs['true_velo_x_val']
    phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']

    print("..................................")
    print("PHX val corr vs true velos (w/o access!):", 
    round(np.corrcoef(true_velos_val.flatten(), phx_val_set_pred.flatten())[0,1], 4))
    print("..................................")
    
    X_full = dynamo_vf_inputs['x_full']
    true_velos_full = dynamo_vf_inputs['true_velo_x_full']

    #quit()
    best_val_corr = 0
    best_n_trees = None
    best_rf = None



    
    for this_num_trees in [50, 100, 200, 500, 1000, 2000]: #100, 250, 500, 1000,
        time_start = time.time()
        print("RNA ODE, num_trees:", this_num_trees)
        rf_mod, train_score, test_score = BUILD_MODEL(counts = X_full, 
                                                        velocity = true_velos_full,
                                                        n_estimators=this_num_trees, 
                                                        max_depth=None, 
                                                        train_size=0.9)
        if test_score > best_val_corr:
            print("updating best RF!")
            best_val_corr = test_score 
            best_n_trees = this_num_trees
            best_rf = rf_mod
        
        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))
        print("..................................")

    
    print("Best num trees:", best_n_trees ,"best val corr:", best_val_corr)
    
    print("obtaining GRN now..\n")
    my_GRN = GET_GRN(counts = X_full, velocity = true_velos_full, model_to_test = best_rf)
    np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", my_GRN, delimiter=",")

