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
import matplotlib.pyplot as plt

torch.set_num_threads(4) #CHANGE THIS!
    
sums_model = torch.load('C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_sums.pt')
#alpha = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_model_weights.pt')
gene_mult = torch.load('C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_gene_multipliers.pt')

W1 = np.transpose(sums_model.linear_1.weight.detach().numpy())
B1 = np.transpose(sums_model.linear_1.bias.detach().numpy())
Wo = np.transpose(sums_model.linear_out.weight.detach().numpy())
Bo = np.transpose(sums_model.linear_out.bias.detach().numpy())
#alpha = np.transpose(torch.sigmoid(alpha.detach()).numpy())
gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())


np.savetxt("C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/model_inspect/w1.csv", W1, delimiter=",")
np.savetxt("C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/model_inspect/b1.csv", B1, delimiter=",")
np.savetxt("C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/model_inspect/wo.csv", Wo, delimiter=",")
np.savetxt("C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/model_inspect/bo.csv", Bo, delimiter=",")
#np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/alpha.csv", alpha, delimiter=",")
np.savetxt("C:/STUDIES/RESEARCH/neural_ODE/ode_net/code/model_inspect/gene_mult.csv", gene_mult, delimiter=",")



#plt.clf()
#fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex = True)

#ax0.hist(sums_model.linear_1.weight.detach().numpy().flatten(), bins = 100)
#ax0.set_title('first layer init')

#ax1.hist(sums_model.linear_out.weight.detach().numpy().flatten(), bins = 100)
#ax1.set_title('out layer init')

#ax0.set_xlim(-0.2, 0.2)
#ax1.set_xlim(-0.2, 0.2)

#plt.savefig('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/viz_NN_weights.png')