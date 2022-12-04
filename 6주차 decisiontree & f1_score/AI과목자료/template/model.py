import torch.nn as nn
class Q_net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        ######################################################
        TODO: Implement Your Model
        ######################################################
        '''   
    def forward(self, x):
        Q_values = self.layers(x)
        return Q_values