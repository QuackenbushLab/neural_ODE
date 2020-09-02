import torch
import torch.nn as nn

class ODENet(nn.Module):
    ''' ODE-Net class implementation '''
    
    def __init__(self, device, ndim):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()
        
        self.ndim = ndim
        # Create a new sequential model with ndim inputs and outputs
        self.net = nn.Sequential(
            nn.Linear(ndim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            #nn.Linear(200, 200),
            #nn.ReLU(),
            nn.Linear(50, ndim ** 2 + ndim)
        )

        # Initialize the layers of the model
        for n in self.net.modules():
            if isinstance(n, nn.Linear):
                nn.init.normal_(n.weight, mean=0, std=.05)
                #nn.init.zeros_(n.weight)
                nn.init.constant_(n.bias, val=0)
                n.to(device)
        
        #self.net.to(device)

    def forward(self, t, y):
        ''' Forward prop through the network '''
        mat = self.net(y).flatten()
        A = mat[0:self.ndim**2]
        A = A.reshape((self.ndim, self.ndim))
        B = mat[self.ndim**2::]
        res = torch.matmul(A, torch.transpose(y, 1, 0)) + B.unsqueeze(1)
        return torch.transpose(res, 1, 0)

    def get_mat(self, t, y):
        mat = self.net(y).flatten()
        A = mat[0:self.ndim**2]
        A = A.reshape((self.ndim, self.ndim))
        B = mat[self.ndim**2::]
        return A, B

    def save(self, fp):
        ''' Save the model to file '''
        torch.save(self.net.state_dict(), fp)

    def load(self, fp):
        ''' Load a model from file '''
        self.net.load_state_dict(torch.load(fp))
        print(self.net.eval())
