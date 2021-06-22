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
    
    def __init__(self, device, ndim, explicit_time=False, neurons=100):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()

        self.ndim = ndim
        self.explicit_time = explicit_time
        
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
            self.net = nn.Sequential()
            #self.net.add_module('activation_0',nn.Softsign())
            self.net.add_module('linear_1', nn.Linear(ndim, neurons))
            self.net.add_module('activation_1',nn.Softplus())
            self.net.add_module('linear_out', nn.Linear(neurons, ndim - self.num_tf))
            
        # Initialize the layers of the model
        for n in self.net.modules():
            if isinstance(n, nn.Linear):
                #nn.init.xavier_normal_(n.weight)
                #nn.init.xavier_normal_(n.bias)
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('sigmoid')) #IH changed init scheme
                #nn.init.constant_(n.bias, val=1)
        
        #self.net.linear_out.weight.data.fill_(-1) #trying this out
        #self.net.linear_out.weight.requires_grad = False #trying this out
        
        self.net.to(device)

    def forward(self, t, y):
        ''' Forward prop through the network'''
        final = torch.zeros(y.shape)
        grad = self.net(y) #subsetting the last dimension [...,0:self.num_tf]
        transformed = torch.exp(grad-y[...,self.num_tf:]) -1
        final[...,self.num_tf:] = transformed
        return(final) 
        
    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        dict_path = fp[:idx] + '_dict' + fp[idx:]
        torch.save(self.net, fp)
        torch.save(self.net.state_dict(), dict_path)

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