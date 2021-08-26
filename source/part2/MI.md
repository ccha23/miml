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

# Mutual Information Estimation

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
from dv import *
import matplotlib.pyplot as plt
from IPython import display
import ipywidgets as widgets
import pandas as pd
import tensorboard as tb

%load_ext tensorboard
%matplotlib inline
```

**How to estimate MI via KL divergence?**

+++

In this notebook, we will introduce a few methods of estimating the mutual information via KL divergence.

+++

We first introduce the Mutual Information Neural Estimation (MINE) method in {cite}`belghazi2018mine`.

+++

## MINE

+++

The idea is to obtain MI {eq}`MI` from KL divergence {eq}`D` as follows:

+++

$$
\begin{align*}
I(\R{X}\wedge \R{Y}) = D(\underbrace{P_{\R{X},\R{Y}}}_{P_{\R{Z}}}\| \underbrace{P_{\R{X}}\times P_{\R{Y}}}_{P_{\R{Z}'}}).
\end{align*}
$$

+++

and then apply the DV formula {eq}`avg-DV` to estimate the divergence:

+++

````{prf:definition} MINE  
:label: MINE

The mutual information neural estimation (MINE) of $I(\R{X}\wedge\R{Y})$ is

$$
\begin{align}
\R{I}_{\text{MINE}} := \sup_{t_{\theta}: \mc{Z} \to \mathbb{R}} \overbrace{\frac1n \sum_{i\in [n]} t_{\theta}(\R{X}_i,\R{Y}_i) - \frac1{n'}\sum_{i\in [n']} e^{t_{\theta}(\R{X}'_i,\R{Y}'_i)}}^{-\R{L}_{\text{MINE}}(\theta):=}
\end{align}
$$ (MINE)

where 

- the supremum is over $t_{\theta}$ representable by a neural network with trainable/optimizable parameters $\theta$,
- $P_{\R{X}',\R{Y}'}:=P_{\R{X}}\times P_{\R{Y}}$, and
- $(\R{X}'_i,\R{Y}'_i\mid i\in [n'])$ is the sequence of i.i.d. samples of $P_{\R{X}',\R{Y}'}$.

````

+++

The above actually does not completely define MINE. There are some implementation details to be filled in.

+++ {"slideshow": {"slide_type": "subslide"}, "tags": []}

**How to obtain the reference samples ${\R{Z}'}^{n'}$, i.e., ${\R{X}'}^{n'}$ and ${\R{Y}'}^{n'}$?**

+++

We can approximate the i.i.d. sampling of $P_{\R{X}}\times P_{\R{Y}}$ using samples from $P_{\R{X},\R{Y}}$ by a re-sampling trick:

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

$$
\begin{align}
P_{\R{Z}'^{n'}} &\approx P_{((\R{X}_{\R{J}_i},\R{Y}_{\R{K}_i})\mid i \in [n'])}
\end{align}
$$ (resample)

where $\R{J}_i$ and $\R{K}_i$ for $i\in [n']$ are independent and uniformly random indices

$$
P_{\R{J},\R{K}} = \op{Uniform}_{[n]\times [n]}
$$

and $[n]:=\Set{1,\dots,n}$.

+++ {"slideshow": {"slide_type": "subslide"}, "tags": []}

MINE {cite}`belghazi2018mine` uses the following implementation that samples $(\R{J},\R{K})$ but without replacement. You can change $n'$ using the slider for `n_`.

```{code-cell} ipython3
rng = np.random.default_rng(SEED)


def resample(data, size, replace=False):
    index = rng.choice(range(data.shape[0]), size=size, replace=replace)
    return data[index]


@widgets.interact
def plot_resampled_data_without_replacement(n_=(2, n)):
    XY_ = np.block([resample(XY[:, [0]], n_), resample(XY[:, [1]], n_)])
    resampled_data = pd.DataFrame(XY_, columns=["X'", "Y'"])
    p_ = plot_samples_with_kde(resampled_data)
    plt.show()
```

In the above, we defined the function `resample` that 
- uses `choice` to uniformly randomly select a choice
- from a `range` of integers going from `0` to 
- the size of the first dimension of the `data`.

+++

Note however that the sampling is without replacement.

+++

**Is it a good idea to sample without replacement?**

+++

**Exercise**

Sampling without replacement has an important restriction $n'\leq n$. Why?

+++

````{toggle}
**Solution**

To allow $n>n'$, at least one sample $(\R{X}_i,\R{Y}_i)$ needs to be sampled more than once.

````

+++

**Exercise** To allow $n>n'$, complete the following code to sample with replacement and observe what happens when $n \gg n'$.

```{code-cell} ipython3
:tags: [hide-cell]

@widgets.interact
def plot_resampled_data_with_replacement(
    n_=widgets.IntSlider(n, 2, 10 * n, continuous_update=False)
):
    ### BEGIN SOLUTION
    XY_ = np.block(
        [resample(XY[:, [0]], n_, replace=True), resample(XY[:, [1]], n_, replace=True)]
    )
    ### END SOLUTION
    resampled_data = pd.DataFrame(XY_, columns=["X'", "Y'"])
    p_ = plot_samples_with_kde(resampled_data)
    plt.show()
```

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

**Exercise** 

Explain whether the resampling trick gives i.i.d. samples $(\R{X}_{\R{J}_i},\R{Y}_{\R{K}_i})$ for the cases with replacement and without replacement respectively?

+++ {"slideshow": {"slide_type": "fragment"}, "tags": ["hide-cell"]}

````{toggle}
**Solution** 

The samples are identically distributed. However, they are not independent except in the trivial case $n=1$ or $n'=1$, regardless of whether the sample is with replacement or not. Consider $n=1$ and $n'=2$ as an example.

````

+++

To improve the stability of the training, MINE applies additional smoothing to the gradient calculation:

+++

$$
\begin{align}
\R{L}_{\text{MINE}}(\theta) &= \overbrace{- \frac{1}{n} \sum_{i\in [n]}  t_{\theta} (\R{X}_i, \R{Y}_i) }^{\R{L}_1(\theta):=} + \log \overbrace{\frac{1}{n'} \sum_{i\in [n']}  e^{t_{\theta} (\R{X}'_i, \R{Y}'_i)}}^{\R{L}_2(\theta):=}\\
\nabla \R{L}_{\text{MINE}}(\theta) &= \nabla \R{L}_1(\theta) + \frac{\nabla \R{L}_2(\theta)}{\R{L}_2(\theta)}
\end{align}
$$

+++

Variation in $\nabla \R{L}_2(\theta)$ leads to the variation of the overall gradient especially when $\R{L}_2(\theta)$ is small. With minibatch gradient descent, the sample average is over a small batch and so the variance can be quite large.

+++

To alleviate such variation, MINE replaces the denominator $\R{L}_2(\theta)$ by its moving average:

+++

$$
\theta^{(i+1)} := \theta^{(i)} - s^{(i)} \nabla \R{L}_1 (\theta^{(i)}) + \frac{\nabla \R{L}_2(\theta^{(i)})}{\overline{\R{L}}_2^{(i)}}
$$

where

$$
\overline{\R{L}}_2^{(i)} =  \beta \overline{\R{L}}_2^{(i-1)} + (1-\beta) \R{L}_2(\theta^{(i)})
$$

for some smoothing factor $\beta\in [0,1]$.

+++

**Exercise**

Implement a neural network trainer for MINE similar to `DVTrainer`. 

```{code-cell} ipython3
:tags: []

DVTrainer??
```

+++ {"tags": []}

## MI-NEE

+++

**Is it possible to generate i.i.d. samples for ${\R{Z}'}^{n'}$?**

+++

Consider another formula for mutual information:

+++

````{prf:proposition}  
:label: MI-3D

$$
\begin{align}
I(\R{X}\wedge \R{Y}) &= D(P_{\R{X},\R{Y}}\|P_{\R{X}'}\times P_{\R{Y}'}) - D(P_{\R{X}}\|P_{\R{X}'}) - D(P_{\R{Y}}\|P_{\R{Y}'})
\end{align}
$$ (MI-3D)

for any product reference distribution $P_{\R{X}'}\times P_{\R{Y}'}$ for which the divergences are finite.

````

+++

````{prf:corollary}  
:label: MI-ub


$$
\begin{align}
I(\R{X}\wedge \R{Y}) &= \inf_{\substack{P_{\R{X}'}\in \mc{P}(\mc{X})\\ P_{\R{Y}'}\in \mc{P}(\mc{Y})}} D(P_{\R{X},\R{Y}}\|P_{\R{X}'}\times P_{\R{Y}'}).
\end{align}
$$ (MI-ub)

where the optimal solution is $P_{\R{X}'}\times P_{\R{Y}'}=P_{\R{X}}\times P_{\R{Y}}$, the product of marginal distributions of $\R{X}$ and $\R{Y}$. 

````

+++

````{prf:proof}

{eq}`MI-ub` follows from {eq}`MI-3D` directly since the divergences are non-negative. To prove the proposition:

$$
\begin{align}
I(\R{X}\wedge \R{Y}) &= H(\R{X}) + H(\R{Y}) - H(\R{X},\R{Y})\\
&= E\left[-\log dP_{\R{X}'}(\R{X})\right] - D(P_{\R{X}}\|P_{\R{X}'})\\
&\quad+E\left[-\log dP_{\R{Y}'}(\R{Y})\right] - D(P_{\R{Y}}\|P_{\R{Y}'})\\
&\quad-E\left[-\log d(P_{\R{X}'}\times P_{\R{Y}'})(\R{X},\R{Y})\right] + D(P_{\R{X},\R{Y}}\|P_{\R{X}'}\times P_{\R{Y}'})\\
&= D(P_{\R{X},\R{Y}}\|P_{\R{X}'}\times P_{\R{Y}'}) - D(P_{\R{X}}\|P_{\R{X}'}) - D(P_{\R{Y}}\|P_{\R{Y}'})
\end{align}
$$

````

+++

*Mutual Information Neural Entropic Estimation (MI-NEE)* {cite}`chan2019neural` uses {eq}`MI-3D` to estimate MI by estimating the three divergences.

+++

Applying {eq}`avg-DV` to each divergence in {eq}`MI-3D`:

+++

$$
\begin{align}
I(\R{X}\wedge \R{Y}) &\approx \sup_{t: \mc{Z} \to \mathbb{R}} \frac1n \sum_{i\in [n]} t_{\R{X},\R{Y}}(\R{X}_i,\R{Y}_i) - \frac1{n'}\sum_{i\in [n']} e^{t_{\R{X},\R{Y}}(\R{X}'_i,\R{Y}'_i)}\\
&\quad - \sup_{t: \mc{Z} \to \mathbb{R}} \frac1n \sum_{i\in [n]} t_{\R{X}}(\R{X}_i) - \frac1{n'}\sum_{i\in [n']} e^{t_{\R{X}}(\R{X}'_i)} \\
&\quad - \sup_{t: \mc{Z} \to \mathbb{R}} \frac1n \sum_{i\in [n]} t_{\R{Y}}(\R{Y}_i) - \frac1{n'}\sum_{i\in [n']} e^{t_{\R{Y}}(\R{Y}'_i)}
\end{align}
$$ (MI-NEE)

+++

$P_{\R{X}'}$ and $P_{\R{Y}'}$ are known distributions and so arbitrarily many i.i.d. samples can be drawn from them directly without using the resampling trick {eq}`resample`.

+++

Indeed, since the choice of $P_{\R{X}'}$ and $P_{\R{Y}'}$ are arbitrary, we can also also train neural networks to optimize them. In particular, {eq}`MI-ub` is a special case where we can train neural networks to approximate $P_{\R{X}}$ and $P_{\R{Y}}$.
