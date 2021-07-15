import torch
import torch.nn as nn
import sys
#torch.set_num_threads(36)

class Expo(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        ex = torch.exp(input)
        return(ex)


class LogX(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        ex = torch.log(input)
        return(ex)

class Recipro(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        ex = torch.reciprocal(input)
        return(ex)        


       
class ODENet(nn.Module):
    ''' ODE-Net class implementation '''
    
    def __init__(self, device, ndim, explicit_time=False, neurons=100, log_scale = "linear", init_bias_y = 0):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()

        self.ndim = ndim
        self.explicit_time = explicit_time
        self.log_scale = log_scale
        self.init_bias_y = init_bias_y
        #only use first 68 (i.e. TFs) as NN inputs
        #in general should be num_tf = ndim
        self.num_tf = 73 
        
        # Create a new sequential model with ndim inputs and outputs
        if explicit_time:
            self.net = nn.Sequential(
                nn.Linear(ndim + 1, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, ndim)
            )
        else: #6 layers
            #self.net_prods = nn.Sequential() #feed log transformed data into this (insize = ndim/genes)
            #self.net_prods.add_module('linear_1', nn.Linear(ndim, neurons))
            #self.net_prods.add_module('activation_1',nn.Softplus(beta = -1))  
            #self.net_prods.add_module('linear_out', nn.Linear(neurons, ndim)) 
            
           # self.prod_signs  = nn.Parameter(4*(torch.rand(1,ndim)-0.5), requires_grad= True)
            
            
            self.net_sums = nn.Sequential()
            self.net_sums.add_module('activation_0',nn.Softsign())
            self.net_sums.add_module('linear_1', nn.Linear(ndim, neurons))
            self.net_sums.add_module('activation_1',nn.Softsign())
            self.net_sums.add_module('linear_out', nn.Linear(neurons, ndim))

            #self.alpha = nn.Parameter(torch.rand(1,ndim), requires_grad= True)
            self.gene_multipliers = nn.Parameter(torch.rand(1,ndim), requires_grad= True)
            #self.model_weights  = nn.Parameter(4*(torch.rand(1,ndim)-0.5), requires_grad= True)
                
        # Initialize the layers of the model
        #for n in self.net_prods.modules():
        #    if isinstance(n, nn.Linear):
        #        nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))

        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))

        #for n in self.net_gene_weights.modules():
        #    if isinstance(n, nn.Linear):
        #        nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))                
        
      
        #self.net_prods.to(device)
        self.net_sums.to(device)
        self.gene_multipliers.to(device)
        #self.model_weights.to(device)
        #self.prod_signs.to(device)

       
        
    def forward(self, t, y):
        #eps = 10**-4
        #y = torch.nn.functional.threshold(y, threshold = eps, value = eps)
        #prods = torch.exp(self.net_prods(torch.log(y))) 
        #prods_with_signs = prods * torch.tanh(self.prod_signs)
        sums = self.net_sums(y-self.init_bias_y)
        
        #alpha = torch.sigmoid(self.model_weights)
        #joint =  (1-alpha)*prods + alpha*sums

        final = torch.relu(self.gene_multipliers)*(sums  - y) #
        return(final) 

        #final = torch.zeros(y.shape)
        #grad = self.net(y[...,self.num_tf:]) #subsetting the last dimension [...,0:self.num_tf]
        #transformed = torch.exp(grad-y[...,self.num_tf:]) + torch.exp(-1*y[...,self.num_tf:]) - 1
        #final[...,self.num_tf:] = transformed


    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        dict_path = fp[:idx] + '_dict' + fp[idx:]
        torch.save(self.net_sums, fp)
        torch.save(self.net_sums.state_dict(), dict_path)

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def load_model(self, fp):
        ''' Load a model from a file '''
        self.net = torch.load(fp)
        self.net.to('cpu')

    def load(self, fp):
        ''' General loading from a file '''
        try:
            print('Trying to load model from file= {}'.format(fp))
            self.load_model(fp)
            print('Done')
        except:
            print('Failed! Trying to load parameters from file...')
            try:
                self.load_dict(fp)
                print('Done')
            except:
                print('Failed! Network structure is not correct, cannot load parameters from file, exiting!')
                sys.exit(0)

    def to(self, device):
        self.net.to(device)