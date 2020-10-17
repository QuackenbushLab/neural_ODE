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

''' OLD VALIDATION FUNCTION
def validation(trajectories, data_handler):
    loss = 0 
    for trajectory, data_trajectory in zip(trajectories, data_handler.data_pt):
        loss += torch.mean((torch.abs(trajectory - data_trajectory) ** 2))
    return loss
'''

def plot_MSE(epoch_so_far, training_loss, validation_loss, true_mean_losses, img_save_dir):
    plt.figure()
    plt.plot(range(1, epoch_so_far + 1), training_loss, color = "blue", label = "Training loss")
    if len(validation_loss) > 0:
        plt.plot(range(1, epoch_so_far + 1), validation_loss, color = "red", label = "Validation loss")
    plt.plot(range(1, epoch_so_far + 1), true_mean_losses, color = "green", label = r'True $\mu$ loss')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')
    plt.ylabel("Error (MSE)")
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
        

def validation(odenet, data_handler, method, explicit_time):
    data, t, target, n_val = data_handler.get_validation_set()
    #print(data)
    #print(data.shape)
    #print(n_val)
    with torch.no_grad():
        if explicit_time:
            if data_handler.batch_type == 'batch_time':
                data = torch.cat((data, t[:,0:-1].reshape((t[:,0:-1].shape[0], t[:,0:-1].shape[1], 1))), 2)
            else:
                data = torch.cat((data, t[:,0].reshape((t[:,0].shape[0], 1, 1))), 2)

            if data_handler.batch_type == 'batch_time':
                target = torch.cat((target, t[:,1::].reshape((t[:,1::].shape[0], t[:,1::].shape[1], 1))), 2)
            else:
                target = torch.cat((target, t[:,1].reshape((t[:,1].shape[0], 1, 1))), 2)
        
        predictions = torch.zeros(data.shape).to(data_handler.device)
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point) in enumerate(zip(t, data)):
            # Do prediction
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] #IH comment
            #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

        # Calculate validation loss
        loss = torch.mean((torch.abs(predictions - target) ** 2))
    return loss

def true_loss(odenet, data_handler, method):
    data, t, target = data_handler.get_true_mu_set()
    with torch.no_grad():
        predictions = torch.zeros(data.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, data)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] #IH comment
        
        # Calculate true mean loss
        loss = torch.mean((torch.abs(predictions - target) ** 2))
    return loss


def decrease_lr(opt, verbose, one_time_drop = 0):
    for param_group in opt.param_groups:
        if one_time_drop == 0:
            param_group['lr'] = param_group['lr']*settings['dec_lr_factor']
        else:
            param_group['lr'] = one_time_drop
    if verbose:
        print("Decreasing learning rate to: %f" % opt.param_groups[0]['lr'])

def training_step(odenet, data_handler, opt, method, batch_size, explicit_time, relative_error):
    batch, t, target = data_handler.get_batch(batch_size)
    opt.zero_grad()

    #print("taking a training step!")
    if explicit_time:
        if data_handler.batch_type == 'batch_time':
            batch = torch.cat((batch, t[:,0:-1].reshape((t[:,0:-1].shape[0], t[:,0:-1].shape[1], 1))), 2)
        else:
            batch = torch.cat((batch, t[:,0:-1].reshape((t[:,0:-1].shape[0], 1, 1))), 2)

        if data_handler.batch_type == 'batch_time':
            target = torch.cat((target, t[:,1::].reshape((t[:,1::].shape[0], t[:,1::].shape[1], 1))), 2)
        else:
            target = torch.cat((target, t[:,1].reshape((t[:,1].shape[0], 1, 1))), 2)
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        # Do prediction and update weights
        predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] #IH comment
        #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

    if relative_error:
        loss = torch.mean((torch.abs((predictions - target)/target) ** 2))
    else:
        loss = torch.mean((torch.abs(predictions - target) ** 2))

    loss.backward()
    opt.step()

    return loss

#def _clean_file_path(fp):
    """Removed folder path and file extension from fp, to be used for saving the model with data name"""
    # Check formatting if on Windows
 #   if os.name == 'nt':
  #      return fp.replace(".\\data\\", '').replace("data/", '').replace(".csv", '')
  #  else:
  #      return fp.replace("data/", '').replace(".csv", '')

def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name = "chalmers_30genes_50samples_0noise"
#parser.add_argument('--data', type=str, default='C:/STUDIES/RESEARCH/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name))
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name))

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
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print("Running on CPU")
        device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = settings['noise'])

    
    # Initialization
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'])
    odenet.float()
    if settings['pretrained_model']:
        pretrained_model_file = '{}/_pretrained_best_model/best_train_model.pt'.format(settings['output_dir'])
        odenet.load(pretrained_model_file)
        #print("Loaded in pre-trained model!")
        
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())

    # Select optimizer
    print('Using optimizer: {}'.format(settings['optimizer']))
    if settings['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        opt = optim.SGD(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'adagrad':
        opt = optim.Adagrad(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    else:
        opt = optim.Adam(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])

    #mu_loss = true_loss(odenet, data_handler, settings['method']) #IH testing!
    
    # Init plot
    if settings['viz']:
        #if cleaned_file_name.startswith('1d_parabolic'):
        #    if settings['explicit_time']:
        #        visualizer = Visualizator1DTimeDependent(data_handler, odenet, settings)
        #    else:
        #        visualizer = Visualizator1D(data_handler, odenet, settings)
        visualizer = Visualizator1D(data_handler, odenet, settings)

    # Training loop
    #batch_times = []
    epoch_times = []
    total_time = 0
    validation_loss = []
    training_loss = []
    true_mean_losses = []
    A_list = []

    min_loss = 0
    if settings['batch_type'] == 'single':
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])
    elif settings['batch_type'] == 'trajectory':
        iterations_in_epoch = data_handler.train_data_length
    else:
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])

    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    start_time = perf_counter()

    
    tot_epochs = settings['epochs']
    viz_epochs = [round(tot_epochs*1/5), round(tot_epochs*2/5), round(tot_epochs*3/5), round(tot_epochs*4/5),tot_epochs]
    rep_epochs = [25, 40, 50, 80, 120, 160, 200, 240, tot_epochs]
    one_time_drop_done = False 

    for epoch in range(1, tot_epochs + 1):
        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        #visualizer.save(img_save_dir, epoch) #IH added to test
        if settings['verbose']:
            print()
            print("[Running epoch {}/{}]".format(epoch, settings['epochs']))
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss: ")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            loss = training_step(odenet, data_handler, opt, settings['method'], settings['batch_size'], settings['explicit_time'], settings['relative_error'])
            #batch_times.append(perf_counter() - start_batch_time)

            # Print and update plots
            iteration_counter += 1
            if settings['verbose']:
                pbar.update(1)
                pbar.set_description("Training loss: {:.5E}".format(loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = loss.item()
        training_loss.append(train_loss)
        mu_loss = true_loss(odenet, data_handler, settings['method'])
        true_mean_losses.append(mu_loss)

        if epoch == 1:
                min_train_loss = train_loss
        else:
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                true_loss_of_min_train_model =  mu_loss
                save_model(odenet, output_root_dir, 'best_train_model')
        
               
        if settings['verbose']:
            pbar.close()

        if settings['solve_A']:
            A = solve_eq(odenet, settings['solve_eq_gridsize'], (-5, 5, 0, 10, -3, 3, -10, 10))
            A_list.append(A)
            print('A =\n{}'.format(A))

        #handle true-mu loss
       
        if data_handler.n_val > 0:
            val_loss = validation(odenet, data_handler, settings['method'], settings['explicit_time'])
            validation_loss.append(val_loss)
            if epoch == 1:
                min_val_loss = val_loss
                print('Model improved, saving current model')
                save_model(odenet, output_root_dir, 'best_val_model')
            else:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss
                    #saving true-mean loss of best val model
                    print('Model improved, saving current model')
                    save_model(odenet, output_root_dir, 'best_val_model')
                    
            print("Validation loss {:.5E}".format(val_loss))
        print("True mu loss {:.5E}".format(mu_loss))

        if settings['viz'] and epoch in viz_epochs:
            print("Saving plot")
            with torch.no_grad():
                visualizer.visualize()
                visualizer.plot()
                visualizer.save(img_save_dir, epoch)
        
        #print("Saving intermediate model")
        #save_model(odenet, intermediate_models_dir, 'model_at_epoch{}'.format(epoch))

        # Decrease learning rate if specified
        if settings['dec_lr'] and epoch % settings['dec_lr'] == 0:
            decrease_lr(opt, settings['verbose'])
        
        # Decrease learning rate as a one-time thing:
        #if train_loss < 5*10**(-5) and one_time_drop_done == False:
        #    decrease_lr(opt, settings['verbose'], one_time_drop= 0.001)
        #    one_time_drop_done = True

        if epoch in rep_epochs:
            print()
            print("Epoch=", epoch)
            print("Time so far= ", (perf_counter() - start_time)/3600, "hrs")
            print("Best training (MSE) so far= ", min_train_loss)
            if data_handler.n_val > 0:
                print("Best validation (MSE) so far = ", min_val_loss.item())
                print("True loss of best validation model (MSE) = ", true_loss_of_min_val_model.item())
            else:
                print("True loss of best training model (MSE) = ", true_loss_of_min_train_model.item())
            print("Saving MSE plot...")
            plot_MSE(epoch, training_loss, validation_loss, true_mean_losses, img_save_dir)    
                
            #print()
    
    total_time = perf_counter() - start_time

    #print("Saving final model")
    #save_model(odenet, output_root_dir, 'final_model')

    print("Saving times")
    #np.savetxt('{}total_time.csv'.format(output_root_dir), [total_time], delimiter=',')
    #np.savetxt('{}batch_times.csv'.format(output_root_dir), batch_times, delimiter=',')
    np.savetxt('{}epoch_times.csv'.format(output_root_dir), epoch_times, delimiter=',')

    if settings['solve_A']:
        np.savetxt('{}final_A.csv'.format(output_root_dir), A, delimiter=',')
        np.save('{}A_at_epochs.npy'.format(output_root_dir), A_list)
        np.save('{}val_loss_at_epochs.npy'.format(output_root_dir), validation_loss)


  
 