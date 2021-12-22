# Imports
import sys
import os
import argparse
from datetime import datetime
import numpy as np
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

torch.set_num_threads(8) #since we are on c5.2xlarge


def my_r_squared(output, target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr**2)

def validation(odenet, data_handler, method, explicit_time):
    data, t, target_full, n_val = data_handler.get_validation_set()

    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point, target_point) in enumerate(zip(t, data, target_full)):
            #IH: 9/10/2021 - added these to handle unequal time availability 
            #comment these out when not requiring nan-value checking
            not_nan_idx = [i for i in range(len(time)) if not torch.isnan(time[i])]
            time = time[not_nan_idx]
            not_nan_idx.pop()
            batch_point = batch_point[not_nan_idx]
            target_point = target_point[not_nan_idx]
            # Do prediction
            predictions.append(odeint(odenet, batch_point, time, method=method)[1])
            targets.append(target_point) #IH comment
            #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

        # Calculate validation loss
        predictions = torch.cat(predictions, dim = 0).to(data_handler.device) #IH addition
        targets = torch.cat(targets, dim = 0).to(data_handler.device) 
        loss = torch.mean(torch.abs((predictions - targets)/targets)) #RELATIVE % error!
        
        
    return [loss, n_val]

def true_loss(odenet, data_handler, method):
    data, t, target = data_handler.get_true_mu_set() #tru_mu_prop = 1 (incorporate later)
    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = torch.zeros(data.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, data)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] + init_bias_y #IH comment
        
        # Calculate true mean loss
        loss =  torch.mean(torch.abs((predictions - target)/target))
        var_explained = my_r_squared(predictions, target)
    return [loss, var_explained]



def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

#def save_model(odenet, folder, filename):
#    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='val_config_inte.cfg')
clean_name =  "desmedt_500genes_1sample_186T" #"
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/{}.csv'.format(clean_name))

args = parser.parse_args()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    cleaned_file_name = "val_only_"+clean_name
    save_file_name = _build_save_file_name(cleaned_file_name, settings['epochs'])

    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)

    img_save_dir = '{}img/'.format(output_root_dir)
    #intermediate_models_dir = '{}intermediate_models/'.format(output_root_dir)

    # Create image and model save directory
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    #if not os.path.exists(intermediate_models_dir):
    #    os.mkdir(intermediate_models_dir)

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
                                        scale_expression = settings['scale_expression'])

    
    # Initialization
    #print(data_handler.dim)
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'])
    odenet.float()
    pretrained_model_file = '/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
    odenet.load(pretrained_model_file)
    
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
    
    print("Loaded in pre-trained model!")
        
    
    
    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, odenet, settings)
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    #val_loss_list = validation(odenet, data_handler, settings['method'], settings['explicit_time'])
    loss_calcs = true_loss(odenet, data_handler, settings['method'])
    
    #print(val_loss_list)
    #print("Validation loss {:.2%}, using {} points".format(val_loss_list[0], val_loss_list[1]))
    print("True loss {:.2%}".format(loss_calcs[0]))
    print("Variance explained {:.2%}".format(loss_calcs[1]))
    
    np.savetxt('{}val_loss.csv'.format(output_root_dir), [loss_calcs], delimiter=',')
    print("DONE!")

  
 