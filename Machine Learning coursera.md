# Machine Learning by Andrew Ng


### Module 1: Introduction

A **machine learning problem** is defined by the following:

T := Task that needs to be solved

E := Experience of many attempts to solve the task

P := The probability that the task will be solved

#### Supervised Learning

Given $X$ and $Y$, we find mapping $f:X\rightarrow Y$

- A problem is a **regression problem** if the domain of $Y$ is **continuous**.
- A problem is a **classification problem** if the domain of $Y$ is **discrete**.

#### Unsupervised Learning

Given $X$ we find $f \ni X \sim f$

### Module 2: Model and Cost function


#### Model Representation

$m$ := the number of training examples

$\textbf{x}$ := input variables or features

$\textbf{y}$ := output variables or targets

$(x,y)$ := a single training example

$h$ := our hypothesis $\ni h: X \rightarrow Y$ where $X$, $Y$ is the domain of $\textbf{x}$, $\textbf{y}$

In our example we will parameterize $h$ in the following way:

\[ h_\theta = \theta_0 + \theta_1x
\]

This is univariate linear regression or linear regression with one variable.

#### Cost Function

We want to make $h$ more accurate thus we want:
\[
\arg\min_{\theta}{C(h_\theta(x),y)}
\]

We usually choose $C(x,y) = \frac{1}{m}\sum{(x-y)^2}$ called least squares.

In machine learning the cost function is expressed in terms of the parameters $\Theta$

\[
J(\theta_0,\theta_1) = \frac{1}{2m}\sum{(h_\theta(x^{(i)})-y^{(i)})^2}
\]

$\frac{1}{2}$ is chosen to multiply with the cost function to make it easier to calculate the gradient since there is a 2 when the derivative is taken of the cost function $J$

We want to choose a cost function that is convex so that an optimal solution exists.

#### Gradient Descent

Repeat until convergence:

$\Theta = \Theta - \alpha \nabla_\Theta J(\Theta)$

Note: Updates to $\Theta$ must be simultaneous.

$\alpha$ := the "learning" rate or step-size of the optimization step

## Terms Appendix

T := Task that needs to be solved

E := Experience of many attempts to solve the task

P := The probability that the task will be solved

$m$ := the number of training examples

$h$ := our hypothesis $\ni h: \textbf{x} \rightarrow \textbf{y}$

$\alpha$ := the "learning" rate or step-size of the optimization step
