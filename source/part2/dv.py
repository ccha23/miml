import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


# neural network
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super().__init__()
        # fully-connected (fc) layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # layer 2
        self.fc3 = nn.Linear(hidden_size, 1)  # layer 3
        nn.init.normal_(self.fc1.weight, std=sigma)  #
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, z):
        a1 = F.elu(self.fc1(z))
        a2 = F.elu(self.fc2(a1))
        t = self.fc3(a2)
        return t


# sample DV lower bound
def DV(Z, Z_ref, net):
    avg_tZ = net(Z).mean()  # (a)
    avg_etZ_ref = net(Z_ref).logsumexp(dim=0) - np.log(Z_ref.shape[0])  # (b) - (c)
    return avg_tZ - avg_etZ_ref


# DV train
class DVTrainer:
    """
    Neural estimator for KL divergence based on the sample DV lower bound.

    Estimate D(P_Z||P_Z') using samples Z and Z' by training a network t to maximize
        avg(t(Z)) - log avg(e^t(Z'))

    parameters:
    ----------

    Z, Z_ref : Tensors with first dimension indicing the samples of Z and Z' respect.
    net : The neural network t that take Z as input and output a real number for each sample.
    n_iters_per_epoch : Number of iterations per epoch.
    writer_params : Parameters to be passed to SummaryWriter for logging.
    """

    # constructor
    def __init__(self, Z, Z_ref, net, n_iters_per_epoch, writer_params={}, **kwargs):
        self.Z = Z
        self.Z_ref = Z_ref
        self.net = net

        # set optimizer
        self.optimizer = optim.Adam(net.parameters(), **kwargs)

        # batch sizes
        self.n_iters_per_epoch = n_iters_per_epoch  # ideally a divisor of both n and n'
        self.batch_size = int((Z.shape[0] + 0.5) / n_iters_per_epoch)
        self.batch_size_ref = int((Z_ref.shape[0] + 0.5) / n_iters_per_epoch)

        # logging
        self.writer = SummaryWriter(
            **writer_params
        )  # create a new folder under runs/ for logging
        self.n_iter = self.n_epoch = 0  # keep counts for logging

    def step(self, epochs=1):
        """
        Carries out the gradient descend for a number of epochs and returns 
        the divergence estimate evaluated over the entire data.

        Loss for each epoch is recorded into the log, but only one divergence 
        estimate is computed/logged using the entire dataset. Rerun the method,
        using a loop, to continue to train the neural network and log the result.

        Parameters:
        ----------
        epochs : number of epochs
        """
        for i in range(epochs):
            self.n_epoch += 1

            # random indices for selecting samples for all batches in one epoch
            idx = torch.randperm(self.Z.shape[0])
            idx_ref = torch.randperm(self.Z_ref.shape[0])

            for j in range(self.n_iters_per_epoch):
                self.n_iter += 1
                self.optimizer.zero_grad()

                # obtain a random batch of samples
                batch_Z = self.Z[idx[i : self.Z.shape[0] : self.batch_size]]
                batch_Z_ref = self.Z_ref[
                    idx_ref[i : self.Z_ref.shape[0] : self.batch_size_ref]
                ]

                # define the loss as negative DV divergence lower bound
                loss = -DV(batch_Z, batch_Z_ref, self.net)
                loss.backward()  # calculate gradient
                self.optimizer.step()  # descend

                self.writer.add_scalar("Loss/train", loss.item(), global_step=self.n_iter)

        with torch.no_grad():
            estimate = DV(Z, Z_ref, self.net).item()
            self.writer.add_scalar("Estimate", estimate, global_step=self.n_epoch)
            return estimate
        
def plot_samples_with_kde(df, **kwargs):
    p = sns.PairGrid(df, **kwargs)
    p.map_lower(sns.scatterplot, s=2)  # scatter plot of samples
    p.map_upper(sns.kdeplot)  # kernel density estimate for pXY
    p.map_diag(sns.kdeplot)  # kde for pX and pY
    return p

SEED = 0

# set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# create samples
XY_rng = np.random.default_rng(SEED)
rho = 1 - 0.19 * XY_rng.random()
mean, cov, n = [0, 0], [[1, rho], [rho, 1]], 1000
XY = XY_rng.multivariate_normal(mean, cov, n)

XY_ref_rng = np.random.default_rng(SEED)
cov_ref, n_ = [[1, 0], [0, 1]], n
XY_ref = XY_ref_rng.multivariate_normal(mean, cov_ref, n_)

Z = Tensor(XY).to(DEVICE)
Z_ref = Tensor(XY_ref).to(DEVICE)

# create a neural network
torch.manual_seed(SEED)
net = Net().to(DEVICE)

ground_truth = -0.5 * np.log(1 - rho ** 2)