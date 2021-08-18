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
device = torch.device("cuda" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def gaussian_data(rho=0.9, mean=[0, 0],sample_size=100):
    cov = np.array([[1, rho], [rho, 1]])
    return np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=sample_size)

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

def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

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


def train_gan_minee(args, X, Y):
    # Initialize generator and discriminator
    generator1 = Generator(dim=args.d, hidden_dim=args.hidden_dim, y_dim=args.d)
    generator2 = Generator(dim=args.d, hidden_dim=args.hidden_dim, y_dim=args.d)
    # discriminator = Net(input_size=d*2, hidden_size=100)
    discriminator = Discriminator(input_size=args.d*2, hidden_size=100)

    if args.load_available and os.path.exists(args.chkpt_name):
        checkpoint = torch.load(args.chkpt_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        mi_list = checkpoint['mi_list']
        discriminator.load_state_dict(checkpoint['discriminator'])
        generator1.load_state_dict(checkpoint['generator1'])
        generator2.load_state_dict(checkpoint['generator2'])
    else:
        mi_list = []
        if cuda:
            generator1.cuda()
            generator2.cuda()
            discriminator.cuda()

    
    # Optimizers Adam
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_G = torch.optim.Adam(
        itertools.chain(generator1.parameters(), generator2.parameters()), lr=args.lr, betas=(args.b1, args.b2)
    )

    for i in tqdm(range(args.n_iters)):
        z = FloatTensor(np.random.normal(0, 1, (args.sample_size*args.ref_batch_factor, args.d)))

        y_gen = generator1(z)
        x_gen = generator2(z)

        XY = torch.cat((X, Y), dim=1)
        y_gen_ref = _resample(y_gen, batch_size=args.batch_size*args.ref_batch_factor)
        x_gen_ref = _resample(x_gen, batch_size=args.batch_size*args.ref_batch_factor)

        batch_XY = _resample(XY, batch_size=args.batch_size)
        batch_XY_gen_ref = torch.cat((x_gen_ref,y_gen_ref), dim=1)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        log_mean_ef_ref = torch.logsumexp(discriminator(batch_XY_gen_ref), 0) - np.log(batch_XY_gen_ref.shape[0])
        gen_loss = -log_mean_ef_ref
        gen_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # define the loss function with moving average in the gradient estimate
        for _ in range(2):
            z = FloatTensor(np.random.normal(0, 1, (args.sample_size*args.ref_batch_factor, args.d)))

            y_gen = generator1(z)
            x_gen = generator2(z)

            XY = torch.cat((X, Y), dim=1)
            y_gen_ref = _resample(y_gen, batch_size=args.batch_size*args.ref_batch_factor)
            x_gen_ref = _resample(x_gen, batch_size=args.batch_size*args.ref_batch_factor)

            batch_XY = _resample(XY, batch_size=args.batch_size)
            batch_XY_gen_ref = torch.cat((x_gen_ref,y_gen_ref), dim=1)

            optimizer_D.zero_grad()
            mean_fXY = discriminator(batch_XY).mean()
            mean_efXY_ref = torch.exp(discriminator(batch_XY_gen_ref)).mean()
            args.ma_ef = (1-args.ma_rate)*args.ma_ef + args.ma_rate*mean_efXY_ref
            batch_loss_XY = - mean_fXY + (1 / args.ma_ef).detach() * mean_efXY_ref
            batch_loss_XY.backward()
            optimizer_D.step()

        mi_list.append(div(discriminator, XY, batch_XY_gen_ref).cpu().item())
    torch.save({
        'mi_list': mi_list,
        'discriminator': discriminator.state_dict(),
        'generator1': generator1.state_dict(), 
        'generator2': generator2.state_dict()
    }, args.chkpt_name)
    
    # torch.save(discriminator.state_dict(), args.chkpt_name_disc)
    # torch.save(generator1.state_dict(), args.chkpt_name_gen1)
    # torch.save(generator2.state_dict(), args.chkpt_name_gen2)
    return mi_list

def smooth_ce_loss(pre_label, true_label, smoothing, num_classes):
    new_labels = (1.0 - smoothing) * true_label + smoothing / num_classes
    return torch.nn.BCELoss()(pre_label, new_labels)

def acti_func(x, a, b, c):
    # a is \alpha_0, b is \tau and c is 1-\tau in the paper
    alpha = torch.zeros_like(x)
    x_cpu = x.cpu()
    alpha[np.where(x_cpu.cpu()<=b)] = - a*x[np.where(x_cpu<=b)]/b + a
    alpha[np.where((x_cpu>b) & (x_cpu<c))] = 0
    alpha[np.where(x_cpu>=c)] = a*x[np.where(x_cpu>=c)]/(1-c) + a*c/(c-1)
    return alpha

def train_classifier(args, X, Y, loss_type):
    model = Net(input_size=X.shape[1], hidden_size=args.hidden_size)
    if args.load_available and os.path.exists(args.chkpt_name):
        model.load_state_dict(torch.load(args.chkpt_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if cuda:
        model.to(device)
    # cross-entropy loss
    for i in range(args.niters):
        batch_X, batch_Y = resample(X, Y, batch_size=args.batch_size, replace=True)
        optimizer.zero_grad()
        pred_Y = model(batch_X)
        acc = accuracy(batch_Y.reshape(-1,1), pred_Y)
        if loss_type == "ce_loss":
            loss = torch.nn.BCELoss()(pred_Y, batch_Y.reshape(-1,1))
        elif loss_type == "ls_loss":
            loss = smooth_ce_loss(pred_Y, batch_Y.reshape(-1,1), 0.1, 2)
        elif loss_type == "AdapLS_loss":
            a, b, c = 0.01, 1e-3, 1-1e-3
            conf = acti_func(pred_Y, a, b, c)
            loss = smooth_ce_loss(pred_Y, batch_Y.reshape(-1,1), conf.detach(), 2)
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        if i%1000==0:
            print("Iternation: %d, loss: %f, acc: %f"%(i, loss.item(), acc))
    torch.save(model.state_dict(), args.chkpt_name)
    return model