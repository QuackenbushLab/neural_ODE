import torch
import torch.nn as nn
import sys

from torch.nn.init import calculate_gain
#torch.set_num_threads(36)

def off_diag_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, LogSigProdLayer):
        with torch.no_grad():
            m.weight.copy_(torch.triu(m.weight, diagonal = 1) + torch.tril(m.weight, diagonal = -1))

def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook    


class SoftsignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        shifted_input = 50*input - 25 #torch.exp(input -10)
        abs_shifted_input = torch.abs(shifted_input)
        return(shifted_input/(1+abs_shifted_input))  

class LogShiftedSoftSignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        shifted_input =  input -0
        abs_shifted_input = torch.abs(shifted_input)
        soft_sign_mod = shifted_input/(1+abs_shifted_input)
        return(torch.log1p(soft_sign_mod))  


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
           
            #self.net_prods = nn.Sequential()
            #self.net_prods.add_module('activation_0',  LogShiftedSoftSignMod()) #
            #self.net_prods.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))
            
            self.net_sums = nn.Sequential()
            self.net_sums.add_module('activation_0', SoftsignMod())
            self.net_sums.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))

            self.net_alpha_combine = nn.Sequential()
            self.net_alpha_combine.add_module('linear_out',nn.Linear(neurons, ndim, bias = False))
          
            #self.gene_multipliers = nn.Sequential()
            #self.gene_multipliers.add_module('linear_1', nn.Linear(ndim, 25, bias = False))
            #self.gene_multipliers.add_module('linear_out', nn.Linear(25, ndim, bias = False))
            #self.gene_multipliers.add_module('activation_out', nn.ReLU())
  
            self.gene_multipliers = nn.Parameter(torch.rand(1,ndim)*0.5, requires_grad= True)
            #self.gene_taus = nn.Parameter(torch.randn(1,ndim, requires_grad= True))
            #print("mult_mean =", torch.mean(torch.relu(self.gene_multipliers)))
            #print("mult_min =", torch.min(torch.relu(self.gene_multipliers)))
            #print("mult_max =", torch.max(torch.relu(self.gene_multipliers)))
            
            
            #self.minus_effect_factor = nn.Parameter(torch.zeros(1)+3, requires_grad= False) 
                
        # Initialize the layers of the model
        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                #nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)   #0.05  

        #for n in self.gene_multipliers.modules():
        #    if isinstance(n, nn.Linear):
        #        nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
        
        #for n in self.net_prods.modules():
        #    if isinstance(n, nn.Linear):
        #        nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05) #0.05
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                
        for n in self.net_alpha_combine.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                #nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)
                
        #self.net_prods.apply(off_diag_init)
        #self.net_sums.apply(off_diag_init)
        
      
        #creating masks and register the hooks
        #mask_prods = torch.tril(torch.ones_like(self.net_prods.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_prods.linear_out.weight), diagonal = 1)
        #mask_sums = torch.tril(torch.ones_like(self.net_sums.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_sums.linear_out.weight), diagonal = 1)
        
        #self.net_prods.linear_out.weight.register_hook(get_zero_grad_hook(mask_prods))
        #self.net_sums.linear_out.weight.register_hook(get_zero_grad_hook(mask_sums)) 

        
        #self.net_prods.to(device)
        self.gene_multipliers.to(device)
        #self.gene_taus.to(device)
        self.net_sums.to(device)
        self.net_alpha_combine.to(device)
        #self.minus_effect_factor.to(device)
       
        
    def forward(self, t, y):
        sums = self.net_sums(y)
        #prods_part = self.net_prods(y)
        #if torch.any(torch.isnan(prods_part)):
        #    print("we got prods problems!")
        #    print(torch.topk(y, 20, largest = False))
        #prods = torch.exp(prods_part)
        #sums_prods_concat = torch.cat((sums, prods), dim= - 1)
        #joint = self.net_alpha_combine(sums_prods_concat)
        joint = self.net_alpha_combine(sums)
        carry_cap = torch.sigmoid(joint)
        final = 1/10*y*(carry_cap - y) 
        return(final) 

    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        alpha_comb_path = fp[:idx] + '_alpha_comb' + fp[idx:]
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        model_tau_path = fp[:idx] + '_gene_taus' + fp[idx:]
        #torch.save(self.net_prods, prod_path)
        torch.save(self.net_sums, sum_path)
        torch.save(self.net_alpha_combine, alpha_comb_path)
        torch.save(self.gene_multipliers, gene_mult_path)
        #torch.save(self.gene_taus, model_tau_path)
        

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def load_model(self, fp):
        ''' Load a model from a file '''
        idx = fp.index('.')
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        alpha_comb_path = fp[:idx] + '_alpha_comb' + fp[idx:]
        #self.net_prods = torch.load(prod_path)
        self.net_sums = torch.load(sum_path)
        self.gene_multipliers = torch.load(gene_mult_path)
        self.net_alpha_combine = torch.load(alpha_comb_path)
        
        #self.net_prods.to('cpu')
        self.net_sums.to('cpu')
        self.gene_multipliers.to('cpu')
        self.net_alpha_combine.to('cpu')

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