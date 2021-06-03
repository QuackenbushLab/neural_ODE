import torch
import torch.nn as nn
import sys
#torch.set_num_threads(36)

class ExpoMinus(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        ex = torch.exp(-1*input)
        return(ex)

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
            self.net.add_module('linear_0', nn.Linear(ndim, neurons))
            self.net.add_module('activation_0', Expo())
            self.net.add_module('linear_1', nn.Linear(neurons, neurons))
            self.net.add_module('activation_1', nn.Tanh())
            self.net.add_module('linear_2', nn.Linear(neurons, ndim))
            
        # Initialize the layers of the model
        for n in self.net.modules():
            if isinstance(n, nn.Linear):
                #torch.nn.init.xavier_normal_(n.weight, gain = nn.init.calculate_gain('tanh'))
                nn.init.orthogonal_(n.weight,  gain = nn.init.calculate_gain('tanh')) #IH changed init scheme
                #nn.init.constant_(n.bias, val=1)
        
        #self.net.linear_0.weight.requires_grad = False
        #self.net.linear_0.bias.requires_grad = False
        #self.net.linear_1.weight.requires_grad = False
        #self.net.linear_1.bias.requires_grad = False

        self.net.to(device)

    #print("Using {} threads odenet".format(torch.get_num_threads()))

    def forward(self, t, y):
        #torch.set_num_threads(72)
        ''' Forward prop through the network '''
        grad = self.net(y)
        #print("y: {}, t: {}, grad: {:.20f}\n".format(y, t, grad[0,0]))
        if self.explicit_time:
            try:
                grad = torch.cat((grad, torch.ones((y.shape[0], 1, 1))), 2)
            except:
                grad = torch.cat((grad, torch.ones(1).reshape((1, 1))), 1)
        return grad

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