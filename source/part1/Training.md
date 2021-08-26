---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}}

# Training the Neural Network

+++

$\def\abs#1{\left\lvert #1 \right\rvert}
\def\Set#1{\left\{ #1 \right\}}
\def\mc#1{\mathcal{#1}}
\def\M#1{\boldsymbol{#1}}
\def\R#1{\mathsf{#1}}
\def\RM#1{\boldsymbol{\mathsf{#1}}}
\def\op#1{\operatorname{#1}}
\def\E{\op{E}}
\def\d{\mathrm{\mathstrut d}}$

```{code-cell} ipython3
from gaussian import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb

%load_ext tensorboard
%matplotlib inline
```

We will train a neural network with `torch` and use GPU if available:

```{code-cell} ipython3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":  # print current GPU name if available
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
```

When GPU is available, you can use [GPU dashboards][gpu] on the left to monitor GPU utilizations.

[gpu]: https://github.com/rapidsai/jupyterlab-nvdashboard

+++

![GPU](gpu.dio.svg)

+++

**How to train a neural network by gradient descent?**

+++

We will first consider a simple implementation followed by a more practical implementation.

+++

## A simple implementation of gradient descent

+++

Consider solving for a given $z\in \mathbb{R}$,

$$ \inf_{w\in \mathbb{R}} \overbrace{e^{w\cdot z}}^{L(w):=}.$$

We will train one parameter, namely, $w$, to minimize the loss $L(w)$.

+++

**Exercise** 

What is the solution for $z=-1$?

+++ {"nbgrader": {"grade": true, "grade_id": "eg-min", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution**

With $z=-1$,

$$
L(w) = e^{-w} \geq 0
$$

which is achievable with equality as $w\to \infty$.

````

+++

**How to implement the loss function?**

+++

We will define the loss function using tensors:

```{code-cell} ipython3
z = Tensor([-1]).to(DEVICE)  # default tensor type on a designated device


def L(w):
    return (w * z).exp()


L(float("inf"))
```

The function `L` is vectorized because `Tensor` operations follow the [broadcasting rules of `numpy`](https://numpy.org/doc/stable/user/basics.broadcasting.html):

```{code-cell} ipython3
ww = np.linspace(0, 10, 100)
ax = sns.lineplot(
    x=ww,
    y=L(Tensor(ww).to(DEVICE)).cpu().numpy(),  # convert to numpy array for plotting
)
ax.set(xlabel=r"$w$", title=r"$L(w)=e^{-w}$")
ax.axhline(L(float("inf")), ls="--", c="r")
plt.show()
```

**What is gradient descent?**

+++

A gradient descent algorithm updates the parameter $w$ iteratively starting with some initial $w^{(0)}$:

$$w^{(i+1)} = w^{(i)} - s^{(i)} \nabla L(w^{(i)}) \qquad \text{for }i\geq 0,$$

where $s$ is the *learning rate* (*step size*).

+++

**How to compute the gradient?**

+++

With $w^{(0)}=0$, 

$$\nabla L(w^{(0)}) = \left.-e^{-w}\right|_{w=0}=-1,$$ 

which can be computed using `backward` ([backpropagation][bp]):

[bp]: https://en.wikipedia.org/wiki/Backpropagation

```{code-cell} ipython3
w = Tensor([0]).to(DEVICE).requires_grad_()  # requires gradient calculation for w
L(w).backward()  # calculate the gradient by backpropagation
w.grad
```

Under the hood, the function call `L(w)` 

- not only return the loss function evaluated at `w`, but also
- updates a computational graph for calculating the gradient since `w` `requires_grad_()`.

+++

**How to implement the gradient descent?**

+++

With a learning rate of `0.001`:

```{code-cell} ipython3
for i in range(1000):
    w.grad = None  # zero the gradient to avoid accumulation
    L(w).backward()
    with torch.no_grad():  # updates the weights in place without gradient calculation
        w -= w.grad * 1e-3

print("w:", w.item(), "\nL(w):", L(w).item())
```

**What is `torch.no_grad()`?**

+++

It sets up a context where the computational graph will not be updated. In particular,

```Python
w -= w.grad * 1e-3
```

should not be differentiated in the subsequent calculations of the gradient.

[no_grad]: https://pytorch.org/docs/stable/generated/torch.no_grad.html

+++

**Exercise** 

Repeatedly run the above cell until you get `L(w)` below `0.001`. How large is the value of `w`? What is the limitations of the simple gradient descent algorithm?

+++ {"nbgrader": {"grade": true, "grade_id": "gd-limitations", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}


````{toggle}
**Solution** 

The value of `w` needs to be smaller than `6.9`. The convergence can be slow, especially when the learning rate is small. Also, `w` can be far away from its optimal value even if `L(w)` is close to its minimum.

````

+++

## A practical implementation

+++

For a neural network to approximate a sophisticated function, it should have many parameters (*degrees of freedom*).

+++

**How to define a neural network?**

+++

The following code [defines a simple neural network][define] with 3 fully-connected (fc) hidden layers:

![Neural net](nn.dio.svg)

where 

- $\M{W}_l$ and $\M{b}_l$ are the weight and bias respectively for the linear transformation $\R{W}_l a_l + b_l$ of the $l$-th layer; and
- $\sigma$ for the first 2 hidden layers is an activation function called the [*exponential linear unit (ELU)*](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html).

[define]: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#define-the-network

```{code-cell} ipython3
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # fully-connected (fc) layer
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


torch.manual_seed(SEED)  # seed RNG for PyTorch
net = Net().to(DEVICE)
print(net)
```

The neural network is also a vectorized function. E.g., the following call `net` once to plots the density estimate of all $t(\R{Z}_i)$'s and $t(\R{Z}'_i)$'s.

```{code-cell} ipython3
:tags: []

Z = Tensor(XY).to(DEVICE)
Z_ref = Tensor(XY_ref).to(DEVICE)

tZ = (
    net(torch.cat((Z, Z_ref), dim=0))  # compute t(Z_i)'s and t(Z'_i)
    # output needs to be converted back to an array on CPU for plotting
    .cpu()  # copy back to CPU
    .detach()  # detach from current graph (no gradient calculation)
    .numpy()  # convert output back to numpy
)

tZ_df = pd.DataFrame(data=tZ, columns=["t"])
sns.kdeplot(data=tZ_df, x="t")
plt.show()
```

For 2D sample $(x,y)\in \mc{Z}$, we can plot the neural network $t(x,y)$ as a heatmap:

```{code-cell} ipython3
:tags: []

def plot_net_2(net, xmin=-5, xmax=5, ymin=-5, ymax=5, xgrids=50, ygrids=50, ax=None):
    """Plot a heat map of a neural network net. net can only have two inputs."""
    x, y = np.mgrid[xmin : xmax : xgrids * 1j, ymin : ymax : ygrids * 1j]
    xy = np.concatenate((x[:, :, None], y[:, :, None]), axis=2)
    with torch.no_grad():
        z = net(Tensor(xy).to(DEVICE))[:, :, 0].cpu()
    if ax is None:
        ax = plt.gca()
    im = ax.pcolormesh(x, y, z, cmap="RdBu_r", shading="auto")
    ax.figure.colorbar(im)
    ax.set(xlabel=r"$x$", ylabel=r"$y$", title=r"Heatmap of $t(x,y)$")


plot_net_2(net)
```

**Exercise** 

Why are the values of $t(\R{Z}_i)$'s and $t(\R{Z}'_i)$'s concentrated around $0$?

+++ {"nbgrader": {"grade": true, "grade_id": "init-nn", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution** 

The neural network parameters are all very close to $0$ as we have set a small variance `sigma=0.02` to initialize them randomly. Hence:

- The linear transformation $\M{W}_l (\cdot) + \M{b}_l$ is close to $0$ for when the weight and bias are close to $0$. 
- The ELU activation function $\sigma$ is also close to $0$ if its input is close to $0$.

````

+++

**How to implements the divergence estimate?**

+++

We decompose the approximate divergence lower bound in {eq}`avg-DV` as follows:

+++

$$
\begin{align}
\op{DV}(\R{Z}^n,\R{Z'}^{n'},\theta) &:= \underbrace{\frac1{n} \sum_{i\in [n]} t(\R{Z}_i)}_{\text{(a)}} - \underbrace{\log \frac1{n'} \sum_{i\in [n']} e^{t(\R{Z}'_i)}}_{ \underbrace{\log \sum_{i\in [n']} e^{t(\R{Z}'_i)}}_{\text{(b)}} - \underbrace{\log n'}_{\text{(c)}}} 
\end{align}
$$

where $\theta$ is a tuple of parameters (weights and biases) of the neural network that computes $t$:

$$
\theta := (\M{W}_l,\M{b}_l|l\in [3]).
$$

```{code-cell} ipython3
def DV(Z, Z_ref, net):
    avg_tZ = net(Z).mean()  # (a)
    log_avg_etZ_ref = net(Z_ref).logsumexp(dim=0) - np.log(Z_ref.shape[0])  # (b) - (c)
    return avg_tZ - log_avg_etZ_ref


DV_estimate = DV(Z, Z_ref, net)
```

**Exercise** 

Why is it preferrable to use `logsumexp(dim=0)` instead of `.exp().sum().log()`? Try running

```Python
Tensor([100]).exp().log(), Tensor([100]).logsumexp(0)
```

in a separate console.

+++ {"nbgrader": {"grade": true, "grade_id": "logsumexp", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution** 

`logsumexp(dim=0)` is numerically more stable than `.exp().mean().log()` especially when the output of the exponential function is too large to be represented with the default floating point precision.

````

+++

To calculate the gradient of the divergence estimate with respect to $\theta$:

```{code-cell} ipython3
net.zero_grad()  # zero the gradient values of all neural network parameters
DV(Z, Z_ref, net).backward()  # calculate the gradient
a_param = next(net.parameters())
```

`a_param` is a (module) parameter in $\theta$ retrieved from the parameter iterator `parameters()`.

+++

**Exercise** 

Check that the value of `a_param.grad` is non-zero. Is `a_param` a weight or a bias?

+++ {"nbgrader": {"grade": true, "grade_id": "param", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution** 

It should be the weight matrix $\M{W}_1$ because the shape is `torch.Size([100, 2])`.

````

+++

**How to gradient descend?**

+++

We will use the [*Adam's* gradient descend algorithm][adam] implemented as an optimizer [`optim.Adam`][optimAdam]:

[adam]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#cite_note-Adam2014-28
[optimAdam]: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

```{code-cell} ipython3
net = Net().to(DEVICE)
optimizer = optim.Adam(
    net.parameters(), lr=1e-3
)  # Allow Adam's optimizer to update the neural network parameters
optimizer.step()  # perform one step of the gradient descent
```

To alleviate the problem of overfitting, the gradient is often calculated on randomly chosen batches:

+++

$$
\begin{align}
\R{L}(\theta) := - \bigg[\frac1{\abs{\R{B}}} \sum_{i\in \R{B}} t(\R{Z}_i) - \log \frac1{\abs{\R{B}'}} \sum_{i\in \R{B}'} e^{t(\R{Z}'_i)} - \log \abs{\R{B}'} \bigg],
\end{align}
$$

which is the negative lower bound of the VD formula in {eq}`DV` but on the minibatches 

$$\R{Z}_{\R{B}}:=(\R{Z}_i\mid i\in \R{B})\quad \text{and}\quad \R{Z}'_{\R{B}'}$$

where $\R{B}$ and $\R{B}'$ are uniformly randomly chosen indices from $[n]$ and $[n']$ respectively.

+++

An efficient implementation is to 
- permute the samples first, and then
- partition the samples into batches.

+++

![Minibatch gradient descent](batch.dio.svg)

```{code-cell} ipython3
n_iters_per_epoch = 10  # ideally a divisor of both n and n'
batch_size = int((Z.shape[0] + 0.5) / n_iters_per_epoch)
batch_size_ref = int((Z_ref.shape[0] + 0.5) / n_iters_per_epoch)
```

We will use `tensorboard` to show the training logs.  
Rerun the following to start a new log, for instance, after a change of parameters.

```{code-cell} ipython3
if input('New log?[Y/n] ').lower() != 'n':
    n_iter = n_epoch = 0  # keep counts for logging
    writer = SummaryWriter()  # create a new folder under runs/ for logging
```

The following code carries out Adam's gradient descent on batch loss:

```{code-cell} ipython3
if input("Train? [Y/n]").lower() != "n":
    for i in range(10):  # loop through entire data multiple times
        n_epoch += 1

        # random indices for selecting samples for all batches in one epoch
        idx = torch.randperm(Z.shape[0])
        idx_ref = torch.randperm(Z_ref.shape[0])

        for j in range(n_iters_per_epoch):  # loop through multiple batches
            n_iter += 1
            optimizer.zero_grad()

            # obtain a random batch of samples
            batch_Z = Z[idx[i : Z.shape[0] : batch_size]]
            batch_Z_ref = Z_ref[idx_ref[i : Z_ref.shape[0] : batch_size_ref]]

            # define the loss as negative DV divergence lower bound
            loss = -DV(batch_Z, batch_Z_ref, net)
            loss.backward()  # calculate gradient
            optimizer.step()  # descend

        writer.add_scalar("Loss/train", loss.item(), global_step=n_epoch)

    # Estimate the divergence using all data
    with torch.no_grad():
        estimate = DV(Z, Z_ref, net).item()
        writer.add_scalar("Estimate", estimate, global_step=n_epoch)
        plot_net_2(net)
        print('Divergence estimation:', estimate)
```

Run the following to show the losses and divergence estimate in `tensorboard`. You can rerun the above cell to train the neural network more.

```{code-cell} ipython3
%tensorboard --logdir=runs
```

The ground truth is given by

$$D(P_{\R{Z}}\|P_{\R{Z}'}) = \frac12 \log(1-\rho^2) $$

where $\rho$ is the randomly generated correlation in the previous notebook. 

+++

**Exercise** Compute the ground truth using the formula above.

```{code-cell} ipython3
---
nbgrader:
  grade: false
  grade_id: ground_truth
  locked: false
  schema_version: 3
  solution: true
  task: false
tags: [hide-cell]
---
### BEGIN SOLUTION
ground_truth = -0.5*np.log(1-rho**2)
### END SOLUTION
ground_truth
```

**Exercise** 

See if you can get an estimate close to this value by training the neural network repeatedly.

+++

## Encapsulation

+++

It is a good idea to encapsulate the training by a class, so multiple configurations can be run without infering each other:

```{code-cell} ipython3
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
```

To use the above class to train, we first create an instance:

```{code-cell} ipython3
torch.manual_seed(SEED)
net = Net().to(DEVICE)
trainer = DVTrainer(Z, Z_ref, net, n_iters_per_epoch=10)
```

Next, we run `step` iteractively to train the neural network:

```{code-cell} ipython3
if input("Train? [Y/n]").lower() != "n":
    for i in range(10):
        trainer.step(10)
    plot_net_2(net)
```

```{code-cell} ipython3
%tensorboard --logdir=runs
```

## Clean-up

+++

It is important to release the resources if it is no longer used. You can release the memory or GPU memory by `Kernel->Shut Down Kernel`.

+++

To clear the logs:

```{code-cell} ipython3
if input('Delete logs? [y/N]').lower() == 'y':
    !rm -rf ./runs
```

To kill a tensorboard instance without shutting down the notebook kernel:

```{code-cell} ipython3
tb.notebook.list() # list all the running TensorBoard notebooks.
while (pid := input('pid to kill? (press enter to exit)')):
    !kill {pid}
```
