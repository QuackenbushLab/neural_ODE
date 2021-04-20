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

class LogOnePlusX(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        if any(input <-1):
            print("OMGGGGGG! less than -1")
        ex = torch.log1p(input)
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
            self.net = nn.Sequential(
                nn.Linear(ndim, neurons),
                nn.LayerNorm(neurons, elementwise_affine=False),
                nn.Softplus(),
                
                nn.Linear(neurons, neurons),
                nn.LayerNorm(neurons, elementwise_affine=False),
                nn.ReLU(),

                #nn.Linear(neurons, neurons),
                #nn.LayerNorm(neurons, elementwise_affine=False),
                #nn.Sigmoid(),

                #nn.Linear(neurons, neurons),
                #nn.LayerNorm(neurons, elementwise_affine=False),
                #nn.Tanh(),

                nn.Linear(neurons, ndim)
            )

        # Initialize the layers of the model
        for n in self.net.modules():
            if isinstance(n, nn.Linear):
                nn.init.orthogonal_(n.weight, gain=nn.init.calculate_gain('sigmoid')) #IH changed init scheme
                #nn.init.constant_(n.bias, val=1)
        
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