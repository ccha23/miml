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

# Problem Formulation

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
SEED = 0
```

## Mutual information estimation

+++

**How to formulate the problem of mutual information estimation?**

+++ {"slideshow": {"slide_type": "subslide"}}

The problem of estimating the mutual information is:

+++ {"slideshow": {"slide_type": "-"}}

````{prf:definition} MI Estimation  
:label: MI-estimation

Given $n$ samples

$$(\R{X}_1,\R{Y}_1),\dots, (\R{X}_n,\R{Y}_n) \stackrel{iid}{\sim} P_{\R{X},\R{Y}}\in \mc{P}(\mc{X},\mc{Y})$$ 

i.i.d. drawn from an *unknown* probability measure $P_{\R{X},\R{Y}}$ from the space $\mc{X}\times \mc{Y}$, estimate the *mutual information (MI)*

$$
\begin{align}
I(\R{X}\wedge\R{Y}) &:= E\left[\log \frac{d P_{\R{X},\R{Y}}(\R{X},\R{Y})}{d (P_{\R{X}} \times P_{\R{Y}})(\R{X},\R{Y})} \right].
\end{align}
$$ (MI)

````

+++

Run the following code, which uses `numpy` to 
- generate i.i.d. samples from a multivariate gaussian distribution, and
- store the samples as numpy arrays assigned to `XY`.

```{code-cell} ipython3
# Seeded random number generator for reproducibility
XY_rng = np.random.default_rng(SEED)

# Sampling from an unknown probability measure
rho = 1 - 0.19 * XY_rng.random()
mean, cov, n = [0, 0], [[1, rho], [rho, 1]], 1000
XY = XY_rng.multivariate_normal(mean, cov, n)
plt.scatter(XY[:, 0], XY[:, 1], s=2)
plt.show()
```

See [`multivariate_normal`][mn] and [`scatter`][sc].

[mn]: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
[sc]: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

+++

You can also get help directly in [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html):

+++

- **Docstring**: 
  - Move the cursor to the object and 
    - click `Help->Show Contextual Help` or
    - click `Shift-Tab` if you have limited screen space.

+++

- **Directory**:
  - Right click on a notebook and choose `New Console for Notebook`. 
  - Run `dir(obj)` for a previously defined object `obj` to see the available methods/properties of `obj`.

+++

**Exercise** 

What is unknown about the above sampling distribution?

+++ {"nbgrader": {"grade": true, "grade_id": "unknown-distribution", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}, "tags": ["hide-cell"]}

````{toggle}
**Solution**

The density is

$$
\frac{d P_{\R{X},\R{Y}}}{dxdy} = \mc{N}_{\M{0},\left[\begin{smallmatrix}1 & \rho \\ \rho & 1\end{smallmatrix}\right]}(x,y)
$$

but $\rho$ is unknown (uniformly random over $[0.8,0.99)$).

````

+++

To show the data samples using `pandas`:

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
tags: []
---
XY_df = pd.DataFrame(XY, columns=["X", "Y"])
XY_df
```

To plot the data using `seaborn`:

```{code-cell} ipython3
def plot_samples_with_kde(df, **kwargs):
    p = sns.PairGrid(df, **kwargs)
    p.map_lower(sns.scatterplot, s=2)  # scatter plot of samples
    p.map_upper(sns.kdeplot)  # kernel density estimate for pXY
    p.map_diag(sns.kdeplot)  # kde for pX and pY
    return p


plot_samples_with_kde(XY_df)
plt.show()
```

**Exercise** 

Complete the following code by replacing the blanks `___` so that `XY_ref` stores the i.i.d. samples of $(\R{X}',\R{Y}')$ where $\R{X}'$ and $\R{Y}'$ are zero-mean independent gaussian random variables with unit variance.

```Python
...
cov_ref, n_ = ___, n
XY_ref = XY_ref_rng_ref.___(mean, ___, n_)
...
```

```{code-cell} ipython3
---
nbgrader:
  grade: false
  grade_id: sampling
  locked: false
  schema_version: 3
  solution: true
  task: false
tags: [hide-cell]
---
XY_ref_rng = np.random.default_rng(SEED)
### BEGIN SOLUTION
cov_ref, n_ = [[1, 0], [0, 1]], n
XY_ref = XY_ref_rng.multivariate_normal(mean, cov_ref, n_)
### END SOLUTION
XY_ref_df = pd.DataFrame(XY_ref, columns=["X'", "Y'"])
plot_samples_with_kde(XY_ref_df)
plt.show()
```

+++ {"tags": []}

## Divergence estimation

+++

**Can we generalize the problem further?**

+++ {"slideshow": {"slide_type": "subslide"}, "tags": []}

Estimating MI may be viewed as a special case of the following problem:

+++ {"slideshow": {"slide_type": "-"}}

````{prf:definition} Divergence estimation  
:label: D-estimation

Estimate the KL *divergence*

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) &:= E\left[\log \frac{d P_{\R{Z}}(\R{Z})}{d P_{\R{Z}'}(\R{Z})} \right]
\end{align}
$$ (D)

using 
- a sequence $\R{Z}^n:=(\R{Z}_1,\dots, \R{Z}_n)\sim P_{\R{Z}}^n$ of i.i.d. samples from $P_{\R{Z}}$ if $P_{\R{Z}}$ is unknown, and
- another sequence ${\R{Z}'}^{n'}\sim P_{\R{Z}'}^{n'}$ of i.i.d. samples from $P_{\R{Z}'}$  if $P_{\R{Z}'}$, the *reference measure* of $P_{\R{Z}}$, is also unknown.

````

+++ {"slideshow": {"slide_type": "fragment"}, "tags": []}

**Exercise** 

Although $\R{X}^n$ and $\R{Y}^n$ for MI estimation should have the same length, $\R{Z}^n$ and ${\R{Z}'}^{n'}$ can have different lengths, i.e., $n \not\equiv n'$. Why?

+++ {"nbgrader": {"grade": true, "grade_id": "Z-Z_ref", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}, "slideshow": {"slide_type": "fragment"}, "tags": ["hide-cell"]}

````{toggle}
**Solution** 

The dependency between $\R{Z}$ and $\R{Z}'$ does not affect the divergence.

````

+++ {"slideshow": {"slide_type": "subslide"}, "tags": []}

Regarding the mutual information as a divergence from joint to product distributions, the problem can be further generalized to estimtate other divergences such as the $f$-divergence:

+++

For a strictly convex function $f$ with $f(1)=0$,

$$
\begin{align}
D_f(P_{\R{Z}}\|P_{\R{Z}'}) &:= E\left[ f\left(\frac{d P_{\R{Z}}(\R{Z}')}{d P_{\R{Z}'}(\R{Z}')}\right) \right].
\end{align}
$$ (f-D)

+++

$f$-divergence in {eq}`f-D` reduces to KL divergence when $f=u \log u$:

$$
\begin{align}
E\left[ \frac{d P_{\R{Z}}(\R{Z}')}{d P_{\R{Z}'}(\R{Z}')} \log \frac{d P_{\R{Z}}(\R{Z}')}{d P_{\R{Z}'}(\R{Z}')}  \right] &= \int_{\mc{Z}} \color{gray}{d P_{\R{Z}'}(z)} \cdot \frac{d P_{\R{Z}}(z)}{\color{gray}{d P_{\R{Z}'}(z)}} \log \frac{d P_{\R{Z}}(z)}{d P_{\R{Z}'}(z)}. 
\end{align}
$$

+++

**Exercise**

Show that $D_f(P_{\R{Z}}\|P_{\R{Z}'})\geq 0$ with equality iff $P_{\R{Z}}=P_{\R{Z}'}$ using Jensen's inequality and the properties of $f$.

+++ {"nbgrader": {"grade": true, "grade_id": "divergence", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution**

It is a valid divergence because, by Jensen's inequality,

$$
D_f(P_{\R{Z}}\|P_{\R{Z}'}) \geq  f\bigg( \underbrace{E\left[ \frac{d P_{\R{Z}}(\R{Z}')}{d P_{\R{Z}'}(\R{Z}')} \right]}_{=1}\bigg) = 0
$$

with equality iff $P_{\R{Z}}=P_{\R{Z}'}$.

````

+++

Regarding the divergence as an expectation, it is approximated by the sample average:

$$
\begin{align}
D_f(P_{\R{Z}}\|P_{\R{Z}'}) &\approx 
\frac1n \sum_{i\in [n]} f\left(\frac{d P_{\R{Z}}(\R{Z}'_i)}{d P_{\R{Z}'}(\R{Z}'_i)}\right).
\end{align}
$$ (avg-f-D)

+++

However, this is not a valid estimate because it involves the unknown measures $P_{\R{Z}}$ and $P_{\R{Z}'}$.

+++

One may further estimate the *density ratio*

$$
\begin{align}
z \mapsto \frac{d P_{\R{Z}}(z)}{d P_{\R{Z}'}(z)}
\end{align}
$$ (dP-ratio)

+++

or estimate the density defined respective to some reference measure $\mu$:

$$
\begin{align}
p_{\R{Z}}&:=\frac{dP_{\R{Z}}}{d\mu} \in \mc{P}_{\mu}(\mc{Z}).
\end{align}
$$ (density)
