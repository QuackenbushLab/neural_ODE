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

#torch.set_num_threads(4) #CHANGE THIS!
    
ootb_model = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_ootb.pt')

Wo_1 = np.transpose(ootb_model.linear_1.weight.detach().numpy())
Bo_1 = np.transpose(ootb_model.linear_1.bias.detach().numpy())
Wo_2 = np.transpose(ootb_model.linear_out.weight.detach().numpy())
Bo_2 = np.transpose(ootb_model.linear_out.bias.detach().numpy())

np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/wo_1.csv", Wo_1, delimiter=",")
np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/wo_2.csv", Wo_2, delimiter=",")
np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/bo_1.csv", Bo_1, delimiter=",")
np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/bo_2.csv", Bo_2, delimiter=",")


