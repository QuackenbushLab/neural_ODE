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


class LogSigProdLayer(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(LogSigProdLayer, self).__init__() 
        w_sparsity = 0.95
        weight_init = 1+torch.randn(in_channels, out_channels)*0.1
        weight_init = torch.nn.functional.dropout(weight_init, p=w_sparsity, inplace= False, training = True) * (1-w_sparsity)
        self.weight = nn.Parameter(weight_init, requires_grad=True)
        self.bias = nn.Parameter(torch.randn(out_channels) + 17, requires_grad=True)  #adding a bias
        #print("Using in-channels to initialize bias terms")

    def forward(self, x): 
        eps = 10**-3
        x = torch.nn.functional.relu(x) + eps
        log_diag_x = torch.diag_embed(torch.log(torch.squeeze(x)))
        full_mult_mat = torch.matmul(log_diag_x, self.weight)
        log_sig_mat = torch.nn.functional.logsigmoid(full_mult_mat) - torch.log(torch.zeros(1)+0.5) #to cancel out effect of zeroes
        summed_by_column = torch.matmul(torch.ones(x.shape), log_sig_mat)
        y = (summed_by_column + self.bias)/10000
        return y

class SoftMaxLinear(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(SoftMaxLinear, self).__init__() 
        self.weight = nn.Parameter(torch.rand(in_channels, out_channels), requires_grad=True)
        
    def forward(self, x): 
        soft_max_weight = torch.nn.functional.softmax(self.weight, dim = 0)
        y = torch.matmul(x, soft_max_weight)
        return y 

class SoftsignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        shifted_input = input - 0 
        abs_shifted_input = torch.abs(shifted_input)
        return(shifted_input/(1+abs_shifted_input))  

class LogShiftedSoftSignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        shifted_input =  input - 0 
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
           
            self.net_prods = nn.Sequential()
            self.net_prods.add_module('activation_0',  LogShiftedSoftSignMod()) #
            self.net_prods.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))
            
            self.net_sums = nn.Sequential()
            self.net_sums.add_module('activation_0', SoftsignMod())
            self.net_sums.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))

            self.net_alpha_combine = nn.Sequential()
            self.net_alpha_combine.add_module('linear_out',nn.Linear(2*neurons, ndim, bias = False))
          
            self.gene_multipliers = nn.Parameter(torch.rand(1,ndim, requires_grad= True))
            #self.minus_effect_factor = nn.Parameter(torch.zeros(1)+3, requires_grad= False) 
                
        # Initialize the layers of the model
        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                nn.init.sparse_(n.weight,  sparsity=0.98, std = 0.05)   #0.05  

        for n in self.net_prods.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.98, std = 0.05) #0.05
                #torch.nn.init.normal_(n.bias, mean=2, std=1.0)    #try this out

        for n in self.net_alpha_combine.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
                
        #self.net_prods.apply(off_diag_init)
        #self.net_sums.apply(off_diag_init)
        
      
        #creating masks and register the hooks
        #mask_prods = torch.tril(torch.ones_like(self.net_prods.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_prods.linear_out.weight), diagonal = 1)
        #mask_sums = torch.tril(torch.ones_like(self.net_sums.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_sums.linear_out.weight), diagonal = 1)
        
        #self.net_prods.linear_out.weight.register_hook(get_zero_grad_hook(mask_prods))
        #self.net_sums.linear_out.weight.register_hook(get_zero_grad_hook(mask_sums)) 

        
        self.net_prods.to(device)
        self.gene_multipliers.to(device)
        #self.model_weights.to(device)
        self.net_sums.to(device)
        self.net_alpha_combine.to(device)
        #self.minus_effect_factor.to(device)
       
        
    def forward(self, t, y):
        sums = self.net_sums(y)
        prods = torch.exp(self.net_prods(y))
        sums_prods_concat = torch.cat((sums, prods), dim= - 1)
        joint = self.net_alpha_combine(sums_prods_concat)
        #alpha = torch.sigmoid(self.model_weights)
        #joint =  (1-alpha)*prods + alpha*sums
        final = torch.relu(self.gene_multipliers)*(joint - y ) #joint  - y 
        #final = torch.relu(self.gene_multipliers)*(joint - y ) #joint  - y 
        return(final) 

    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        alpha_comb_path = fp[:idx] + '_alpha_comb' + fp[idx:]
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        #model_weight_path = fp[:idx] + '_model_weights' + fp[idx:]
        torch.save(self.net_prods, prod_path)
        torch.save(self.net_sums, sum_path)
        torch.save(self.net_alpha_combine, alpha_comb_path)
        torch.save(self.gene_multipliers, gene_mult_path)
        #torch.save(self.model_weights, model_weight_path)
        

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
        self.net_prods = torch.load(prod_path)
        self.net_sums = torch.load(sum_path)
        self.gene_multipliers = torch.load(gene_mult_path)
        self.net_alpha_combine = torch.load(alpha_comb_path)
        
        self.net_prods.to('cpu')
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