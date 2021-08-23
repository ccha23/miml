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

+++ {"slideshow": {"slide_type": "slide"}, "tags": []}

# Neural Estimation via DV bound

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

+++

Estimating MI well neither require nor imply the divergence/density to be estimated well. However, 
- MI estimation is often not the end goal, but an objective to train a neural network to return the divergence/density. 
- The features/representations learned by the neural network may be applicable to different downstream inference tasks.

+++ {"tags": []}

## Neural estimation of KL divergence

+++

To explain the idea of neural estimation, consider the following characterization of divergence:

+++

````{prf:proposition}  

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) & =  \sup_{Q\in \mc{P}(\mc{Z})} E \left[ \log \frac{dQ(\R{Z})}{dP_{\R{Z}'}(\R{Z})} \right] 
\end{align}
$$ (D1)

where the unique optimal solution is $Q=P_{\R{Z}}$.

````

+++

{eq}`D1` is {eq}`D` but with $P_{\R{Z}}$ replaced by a parameter $Q$.

- The proposition essentially gives a tight lower bound on KL divergence.
- The unknown distribution is recovered as the optimal solution.

+++

````{prf:proof}

To prove {eq}`D1`,

$$
\begin{align*}
D(P_{\R{Z}}\|P_{\R{Z}'})  &= D(P_{\R{Z}}\|P_{\R{Z}'}) - \inf_{Q\in \mc{P}(\mc{Z})} \underbrace{D(P_{\R{Z}}\|Q)}_{\geq 0 \text{ with equality iff } Q=P_{\R{Z}}\kern-3em} \\
&= \sup_{Q\in \mc{P}(\mc{Z})}  \underbrace{D(P_{\R{Z}}\|P_{\R{Z}'})}_{=E \left[\log  \frac{dP_{\R{Z}}(\R{Z})}{dP_{\R{Z}'}(\R{Z})}\right]} -  \underbrace{D(P_{\R{Z}}\|Q)}_{=E \left[\log \frac{dP_{\R{Z}}(\R{Z})}{dQ(\R{Z})}\right]}\\
&= \sup_{Q\in \mc{P}(\mc{Z})} E \left[\log \frac{dQ(\R{Z})}{dP_{\R{Z}'}(\R{Z})}\right]
\end{align*}
$$

````

+++

The idea of neural estimation is to 

- estimate the expectation in {eq}`D1` by the sample average  

$$
\frac1n \sum_{i\in [n]} \log \underbrace{\frac{dQ(\R{Z}_i)}{dP_{\R{Z}'}(\R{Z}_i)}}_{\text{(*)}},
$$

+++

- use a neural network to compute the density ratio (*), and train the network to maximizes the expectation, e.g., by gradient ascent on the above sample average.

+++

Since $Q$ is arbitrary, the sample average above is a valid estimate.

+++

**But how to compute the density ratio?**

+++

We will first consider estimating the KL divergence $D(P_{\R{Z}}\|P_{\R{Z}'})$ when both $P_{\R{Z}}$ and $P_{\R{Z}'}$ are unknown.

+++

## Donsker-Varadhan formula

+++

If $P_{\R{Z}'}$ is unknown, we can apply a change of variable

+++

$$
r(z) = \frac{dQ(z)}{dP_{\R{Z}'}(z)},
$$ (Q->r)

+++

which absorbs the unknown reference into the parameter.

+++

````{prf:proposition}

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) & =  \sup_{\substack{r:\mc{Z}\to \mathbb{R}_+\\ E[r(\R{Z}')]=1}} E \left[ \log r(\R{Z}) \right] 
\end{align}
$$ (D2)

where the optimal $r$ satisfies 
$
r(\R{Z}) = \frac{dP_{\R{Z}}(\R{Z})}{dP_{\R{Z}'}(\R{Z})}.
$ 

````

+++

**Exercise** 

Show using {eq}`Q->r` that the optimal solution satisfies the constraint stated in the supremum {eq}`D2`.

+++ {"nbgrader": {"grade": true, "grade_id": "optional-r", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution**

The constraint on $r$ is obtained from the constraint on $Q\in \mc{P}(\mc{Z})$, i.e., with $dQ(z)=r(z)dP_{\R{Z}'}(z)$, 

$$
\begin{align*}
dQ(z) \geq 0 &\iff r(z)\geq 0\\
\int_{\mc{Z}}dQ(z)=1 &\iff E[r(\R{Z}')]=1.
\end{align*}
$$

````

+++

The next step is to train a neural network that computes $r$. What about?

+++

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) \approx \sup_{\substack{r:\mc{Z}\to \mathbb{R}_+\\ \frac1{n'}\sum_{i\in [n']} r(\R{Z}'_i)=1}} \frac1n \sum_{i\in [n]} \log r(\R{Z}_i)
\end{align}
$$ (avg-D1)

+++

**How to impose the constraint on $r$ when training a neural network?**

+++

We can apply a change of variable:

$$
\begin{align}
r(z)&=\frac{e^{t(z)}}{E[e^{t(\R{Z}')}]}.
\end{align}
$$ (r->t)

+++

**Exercise** 

Show that $r$ defined in {eq}`r->t` satisfies the constraint in {eq}`D1` for all real-valued function $t:\mc{Z}\to \mathbb{R}$.

+++ {"nbgrader": {"grade": true, "grade_id": "r-t", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

````{toggle}
**Solution** 

$$
\begin{align}
E\left[ \frac{e^{t(\R{Z}')}}{E[e^{t(\R{Z}')}]} \right] =  \frac{E\left[ e^{t(\R{Z}')} \right]}{E[e^{t(\R{Z}')}]} = 1.
\end{align}
$$

````

+++

Substituting {eq}`r->t` into {eq}`D1` gives the well-known *Donsker-Varadhan (DV)* formula {cite}`donsker1983asymptotic`:

+++

````{prf:corollary} Donsker-Varadhan

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) =  \sup_{t: \mc{Z} \to \mathbb{R}} E[t(\R{Z})] - \log E[e^{t(\R{Z}')}]
\end{align}
$$ (DV)

where the optimal $t$ satisfies

$$
\begin{align}
t(\R{Z}) = \log \frac{dP_{\R{Z}}(\R{Z})}{dP_{\R{Z}'}(\R{Z})} + c
\end{align}
$$ (DV:sol)

almost surely for some constant $c$.

````

+++

The divergence can be estimated as follows instead of {eq}`avg-D1`:

+++

$$
\begin{align}
D(P_{\R{Z}}\|P_{\R{Z}'}) \approx \sup_{t: \mc{Z} \to \mathbb{R}} \frac1n \sum_{i\in [n]} t(\R{Z}_i) - \frac1{n'}\sum_{i\in [n']} e^{t(\R{Z}'_i)}
\end{align}
$$ (avg-DV)

+++

In summary, the neural estimation of KL divergence is a sample average of {eq}`D` but 

$$
D(P_{\R{Z}}\| P_{\R{Z}'}) = \underset{\stackrel{\uparrow}\sup_t}{} \overbrace{E}^{\op{avg}} \bigg[ \log \underbrace{\frac{P_{\R{Z}}(\R{Z})}{P_{\R{Z}'}(\R{Z})}}_{\frac{e^{t(\R{Z})}}{\underbrace{E}_{\op{avg}}[e^{t(\R{Z}')}]}} \bigg].
$$

but with the unknown density ratio replaced by {eq}`r->t` trained as a neural network.
