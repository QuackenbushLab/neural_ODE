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

torch.set_num_threads(72)

def plot_LR_range_test(all_lrs_used, training_loss, img_save_dir):
    plt.figure()
    plt.plot(all_lrs_used, training_loss, color = "blue", label = "Training loss")
    plt.plot(all_lrs_used, true_mean_losses, color = "green", label = r'True $\mu$ loss')
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Learning rate")
    plt.ylabel("Error (MSE)")
    plt.legend(loc='upper right')
    plt.savefig("{}/LR_range_test.png".format(img_save_dir))

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
        loss = torch.mean((torch.abs(predictions - target) ** 1))
    return [loss, n_val]

def true_loss(odenet, data_handler, method):
    data, t, target = data_handler.get_true_mu_set() #tru_mu_prop = 1 (incorporate later)
    with torch.no_grad():
        predictions = torch.zeros(data.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, data)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] #IH comment
        
        # Calculate true mean loss
        loss = torch.mean((torch.abs(predictions - target) ** 1))
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
    #print("Using {} threads training_step".format(torch.get_num_threads()))
    batch, t, target = data_handler.get_batch(batch_size)
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] #IH comment
    loss = torch.mean((torch.abs(predictions - target) ** 2))
    loss.backward() #MOST EXPENSIVE STEP!
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
clean_name = "chalmers_150genes_100samples_10T_0noise_0bimod"
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
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'])
    odenet.float()
    param_count = sum(p.numel() for p in odenet.parameters() if p.requires_grad)
    param_ratio = round(param_count/ (data_handler.dim)**2, 3)
    print("Using a NN with {} neurons per layer, with {} trainable parameters, i.e. parametrization ratio = {}".format(settings['neurons_per_layer'], param_count, param_ratio))
    
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
    rep_epochs = [5, 15, 25, 40, 50, 80, 120, 160, 200, 240, 300, 350, tot_epochs]
    one_time_drop_done = False 
    rep_epochs_train_losses = []
    rep_epochs_val_losses = []
    rep_epochs_mu_losses = []
    rep_epochs_time_so_far = []
    rep_epochs_so_far = []
    consec_epochs_failed = 0
    epochs_to_fail_to_terminate = 1000
    all_lrs_used = []

    for epoch in range(1, tot_epochs + 1):
        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        #visualizer.save(img_save_dir, epoch) #IH added to test
        this_epoch_total_train_loss = 0
        if settings['verbose']:
            print()
            print("[Running epoch {}/{}]".format(epoch, settings['epochs']))
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss:")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            loss = training_step(odenet, data_handler, opt, settings['method'], settings['batch_size'], settings['explicit_time'], settings['relative_error'])
            #batch_times.append(perf_counter() - start_batch_time)

            this_epoch_total_train_loss += loss.item()
            # Print and update plots
            iteration_counter += 1

            if settings['verbose']:
                pbar.update(1)
                pbar.set_description("Training loss (current batch): {:.5E}".format(loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = this_epoch_total_train_loss/iterations_in_epoch
        training_loss.append(train_loss)
        #print("Overall training loss {:.5E}".format(train_loss))

        mu_loss = true_loss(odenet, data_handler, settings['method'])
        true_mean_losses.append(mu_loss)
        all_lrs_used.append(opt.param_groups[0]['lr'])
        
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
            val_loss_list = validation(odenet, data_handler, settings['method'], settings['explicit_time'])
            val_loss = val_loss_list[0]
            validation_loss.append(val_loss)
            if epoch == 1:
                min_val_loss = val_loss
                print('Model improved, saving current model')
                save_model(odenet, output_root_dir, 'best_val_model')
            else:
                if val_loss < min_val_loss:
                    consec_epochs_failed = 0
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss
                    #saving true-mean loss of best val model
                    print('Model improved, saving current model')
                    save_model(odenet, output_root_dir, 'best_val_model')
                else:
                    consec_epochs_failed = consec_epochs_failed + 1

                    
            print("Validation loss {:.5E}, using {} points".format(val_loss, val_loss_list[1]))
        print("Overall training loss {:.5E}".format(train_loss))
        print("True mu loss {:.5E}".format(mu_loss))

            
        if (settings['viz'] and epoch in viz_epochs) or (settings['viz'] and epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
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
            

        if (epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate) or (val_loss < (0.01 * settings['scale_expression'])**2):
            print()
            rep_epochs_so_far.append(epoch)
            print("Epoch=", epoch)
            rep_time_so_far = (perf_counter() - start_time)/3600
            print("Time so far= ", rep_time_so_far, "hrs")
            rep_epochs_time_so_far.append(rep_time_so_far)
            print("Best training (MSE) so far= ", min_train_loss)
            rep_epochs_train_losses.append(min_train_loss)
            if data_handler.n_val > 0:
                print("Best validation (MSE) so far = ", min_val_loss.item())
                print("True loss of best validation model (MSE) = ", true_loss_of_min_val_model.item())
                rep_epochs_val_losses.append(min_val_loss.item())
                rep_epochs_mu_losses.append(true_loss_of_min_val_model.item())
            else:
                print("True loss of best training model (MSE) = ", true_loss_of_min_train_model.item())
            print("Saving MSE plot...")
            plot_MSE(epoch, training_loss, validation_loss, true_mean_losses, img_save_dir)    
            
            if settings['lr_range_test']:
                plot_LR_range_test(all_lrs_used, training_loss, img_save_dir)

            print("Saving losses")
            if data_handler.n_val > 0:
                L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_val_losses, rep_epochs_mu_losses]
                np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
            #else:
            #    L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_mu_losses]
            #    np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
           
        

        if consec_epochs_failed==epochs_to_fail_to_terminate:
            print("Went {} epochs without improvement; terminating.".format(epochs_to_fail_to_terminate))
            break

        if val_loss < (0.01 * settings['scale_expression'])**2:
            print("SUCCESS! Reached validation target; terminating.")
            break    

    total_time = perf_counter() - start_time

    
    #save_model(odenet, output_root_dir, 'final_model')

    print("Saving times")
    #np.savetxt('{}total_time.csv'.format(output_root_dir), [total_time], delimiter=',')
    #np.savetxt('{}batch_times.csv'.format(output_root_dir), batch_times, delimiter=',')
    np.savetxt('{}epoch_times.csv'.format(output_root_dir), epoch_times, delimiter=',')

    print("DONE!")

  
 