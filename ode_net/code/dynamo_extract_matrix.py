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
    true_velo_pw, velo_t_pw, velo_target_pw = data_handler_velo.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    pred_vels = odenet.forward(t_pw,data_pw)
    
    data_pw = torch.squeeze(data_pw).detach().numpy() #convert to NP array for dynamo VF alg
    true_velo_pw = torch.squeeze(true_velo_pw).detach().numpy() #convert to NP array for dynamo VF alg
    pred_vels = torch.squeeze(pred_vels).detach().numpy() #convert to NP array for dynamo VF alg
    
    print("PHX fitted corr to true velocities (w/o access!):", 
    np.corrcoef(true_velo_pw.flatten(), pred_vels.flatten())[0,1])

    return {'x': data_pw, 'true_velo_x': true_velo_pw, 'phx_velo_x' : pred_vels} 


def training_step(odenet, data_handler, opt, method, batch_size, explicit_time, relative_error, batch_for_prior, prior_grad, loss_lambda):
    #print("Using {} threads training_step".format(torch.get_num_threads()))
    batch, t, target = data_handler.get_batch(batch_size)
    
    '''
    not_nan_idx = [i for i in range(len(t)) if not torch.any(torch.isnan(t[i]))]
    t = t[not_nan_idx]
    batch = batch[not_nan_idx]
    target = target[not_nan_idx]
    '''

    init_bias_y = data_handler.init_bias_y
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + init_bias_y #IH comment
    
    loss_data = torch.mean((predictions - target)**2) 
    
    pred_grad = odenet.prior_only_forward(t,batch_for_prior)
    loss_prior = torch.mean((pred_grad - prior_grad)**2)
    #loss_prior = loss_data

    composed_loss = loss_lambda * loss_data + (1- loss_lambda) * loss_prior
    composed_loss.backward() #MOST EXPENSIVE STEP!
    opt.step()
    return [loss_data, loss_prior]

def _build_save_file_name(save_path, epochs):
    return 'dynamo_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
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
                                        noise = settings['noise'],
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
    X = dynamo_vf_inputs['x']
    obs_velos = dynamo_vf_inputs['true_velo_x']
    my_vf = dyn.vf.SvcVectorField(X = X, V = obs_velos, gamma = 1, M = 50, lambda_ = 3) #gamma = 1 since we dont think there are any outliers
    trained_results = my_vf.train(normalize = False)
    pred_velos = trained_results['V']
    
    corr_coeff = np.corrcoef(obs_velos.flatten(), pred_velos.flatten())[0,1]
    print("The fitted correlation is:", corr_coeff)
    print(trained_results['C'].shape)

    #quit()
    #Jacobian analysis
    jac = my_vf.get_Jacobian()
    n_iter = 50000
    jac_sum = np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    for iter in range(n_iter -1):
        jac_sum = jac_sum + np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    
    jac_avg = np.transpose(jac_sum/n_iter)
    np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", jac_avg, delimiter=",")

    print("")
    print("saved abosulte average jacobian matrix, taken over", n_iter, "random points.")
    quit()
