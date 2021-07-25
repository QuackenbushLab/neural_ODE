import torch
import torch.nn as nn
import sys
#torch.set_num_threads(36)

class SoftsignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        shifted_input = input - 0.5
        abs_shifted_input = torch.abs(shifted_input)
        return(shifted_input/(1+abs_shifted_input))  
'''
class SigmoidShifted(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        shifted_input = input -0.5 #need to figure out the shift
        return(torch.sigmoid(shifted_input))  
'''

class PseudoSquare(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        squared = input*input 
        #squared = torch.relu(1/2 * input) + torch.relu(-1/2 * input) + torch.relu(input - 1/2) + torch.relu(-1*input - 1/2) #approx
        return(squared)  


       
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
            '''
            self.net_prods_act = nn.Sequential() #feed log transformed data into this (insize = ndim/genes)
            self.net_prods_act.add_module('linear_1', nn.Linear(ndim, int(neurons/2)))
            self.net_prods_act.add_module('activation_1',nn.LogSigmoid()) 
            self.net_prods_act.add_module('linear_out', nn.Linear(int(neurons/2), ndim)) 
           
            self.net_prods_rep = nn.Sequential() #feed log transformed data into this (insize = ndim/genes)
            self.net_prods_rep.add_module('linear_1', nn.Linear(ndim, int(neurons/2)))
            self.net_prods_rep.add_module('activation_1',nn.Sigmoid())  
            self.net_prods_rep_2 = nn.Sequential() #feed log transformed data into this (insize = ndim/genes)
            self.net_prods_rep_2.add_module('linear_out', nn.Linear(int(neurons/2), ndim))
            '''    
            
            #self.net_sums = nn.Sequential()
            #self.net_sums.add_module('linear_1', nn.Linear(ndim, neurons))
            #self.net_sums.add_module('activation_1', SigmoidShifted())
            #self.net_sums.add_module('linear_out', nn.Linear(neurons, ndim))

            self.net_prods = nn.Sequential()
            self.net_prods.add_module('activation_0', SoftsignMod())
            self.net_prods.add_module('linear_1', nn.Linear(ndim, neurons))
            self.net_prods.add_module('activation_1', SoftsignMod())
            self.net_prods.add_module('linear_2', nn.Linear(neurons, neurons))
            self.net_prods.add_module('activation_2', PseudoSquare())
            self.net_prods.add_module('linear_out', nn.Linear(neurons, ndim))

          
            self.net_sums = nn.Sequential()
            self.net_sums.add_module('activation_0', SoftsignMod())
            self.net_sums.add_module('linear_1', nn.Linear(ndim, neurons))
            self.net_sums.add_module('activation_1', SoftsignMod())
            self.net_sums.add_module('linear_out', nn.Linear(neurons, ndim))

            #self.alpha = nn.Parameter(torch.rand(1,1), requires_grad= True)
            self.gene_multipliers = nn.Parameter(torch.rand(1,ndim), requires_grad= True)
            self.model_weights  = nn.Parameter(4*(torch.rand(1,ndim)-0.5), requires_grad= True)
                
        # Initialize the layers of the model
        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))

        for n in self.net_prods.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))
                #nn.init.normal_(n.weight, mean = 0.01, std = 0.1)
                #nn.init.normal_(n.bias)
        '''
        for n in self.net_prods_act.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))

        for n in self.net_prods_rep.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))

        for n in self.net_prods_rep_2.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid'))        
        '''
     
        #self.net_prods_act.to(device)
        #self.net_prods_rep.to(device)
        #self.net_prods_rep_2.to(device)
        
        self.net_prods.to(device)
        self.gene_multipliers.to(device)
        self.model_weights.to(device)
        self.net_sums.to(device)

       
        
    def forward(self, t, y):
        #eps = 10**-3
        #y = torch.relu(y) + eps
        #grad_activate = self.net_prods_act(torch.log(y))
        #prods_reppress = torch.log(1-self.net_prods_rep(torch.log(y)))
        #grad_repress = self.net_prods_rep_2(prods_reppress)
        prods = self.net_prods(y)
        #ln_y = -0.693147 + 2*(y-0.5) - 2*(y-0.5)**2 + 2.6667*(y-0.5)**3
        sums = self.net_sums(y)
        
        alpha = torch.sigmoid(self.model_weights)
        joint =  (1-alpha)*prods + alpha*sums

        final = torch.relu(self.gene_multipliers)*(joint  - y) 
        return(final) 

    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        dict_path = fp[:idx] + '_dict' + fp[idx:]
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        model_weight_path = fp[:idx] + '_model_weights' + fp[idx:]
        torch.save(self.net_prods, prod_path)
        torch.save(self.net_sums, sum_path)
        torch.save(self.gene_multipliers, gene_mult_path)
        torch.save(self.model_weights, model_weight_path)
        #torch.save(self.net_prods.state_dict(), dict_path)
        

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def load_model(self, fp):
        ''' Load a model from a file '''
        idx = fp.index('.')
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        model_weight_path = fp[:idx] + '_model_weights' + fp[idx:]
        self.net_prods = torch.load(prod_path)
        self.net_sums = torch.load(sum_path)
        self.gene_multipliers = torch.load(gene_mult_path)
        self.model_weights = torch.load(model_weight_path)
        self.net_prods.to('cpu')
        self.net_sums.to('cpu')
        self.gene_multipliers.to('cpu')

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