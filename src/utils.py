"""
    This is used to store some useful codes which may need in the notebooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import itertools
from tqdm import tqdm

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# Build a neural network
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_layers=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers-1):
            self.fc.append(nn.Linear(hidden_size, hidden_size))
        self.fc.append(nn.Linear(hidden_size,1))
        for i in range(hidden_layers+1):
            nn.init.normal_(self.fc[i].weight, std=sigma)
            nn.init.constant_(self.fc[i].bias, 0)

    def forward(self, input):
        output = input
        for i in range(self.hidden_layers):
            output = F.elu(self.fc[i](output))
        output = torch.sigmoid(self.fc[self.hidden_layers](output))
        return output
        
def resample(X, Y, batch_size, replace=False):
    # Resample a batch of given data samples.
    index = np.random.choice(
        range(X.shape[0]), size=batch_size, replace=replace)
    batch_X = X[index]
    batch_Y = Y[index]
    return batch_X, batch_Y
    
def accuracy(true_labels,pred_labels):  
    return sum((pred_labels > 0.5) == true_labels)/float(true_labels.shape[0])


def uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min

def div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    # log_mean_ef_ref = torch.exp(net(ref)-1).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

class Generator(nn.Module):
    def __init__(self, dim, hidden_dim, y_dim, sigma=0.02):
        super(Generator, self).__init__()
        input_dim = dim
        hidden_size = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, y_dim)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, noise):
        gen_input = noise
        output = F.elu(self.fc1(gen_input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class Discriminator(nn.Module):
    # Inner class that defines the neural network architecture
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output