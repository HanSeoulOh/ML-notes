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


### Module 2: Linear Regression

#### Multivariate Linear Regression


In our example we will parameterize $h$ in the following way:

\[ h_\theta = \Theta^T\textbf{x}
\]

\[ J(\Theta) = \frac{1}{2m}\sum_{i=1}^m(h_\Theta(\textbf{x}^{(i)})-y^{(i)})^2
\]

\[ \nabla_\Theta J(\Theta) = \frac{1}{m}\sum_{i=1}^m(h_\Theta(\textbf{x}^{(i)})-y^{(i)})\nabla_\Theta h_\Theta(\textbf{x}^{(i)})
\]

where $x_0 = 1$ $\forall$ $i$

#### Feature Scaling/Normalization

Usually, features will be adjusted using feature scaling or using the mean normalization method so that gradient descent can converge more quickly by making progress on each feature equally.

Feature scaling is when you let $\textbf{x} = \frac{\textbf{x}}{\max(\textbf{x})-\min(\textbf{x})}$ s.t. $x \in (0,1)$

Mean normalization is when you let $\textbf{x} = \frac{\textbf{x} - \mu(\textbf{x})}{\sigma(\textbf{x})}$

### Module 3: Non-Linear Representation

Using explicit ways to formulate a non-linear hypothesis using polynomial terms can be a very inefficient task. Neural networks are a way to efficiently fit an arbitrary non-linear function to data.

#### Model Representation

We have our bias $x_0$, and inputs $(x_1, ..., x_n)^T$ we define Layer 1 as $X = (x_0, x_1, ..., x_n)^T$

We define Layer 1 as: $A^{(i)} = g(X^T\Theta^{(1)})$. Layer $i$ where $i > 1$ as: $A^{(i)} = g(g(A^{(i-1)T}\Theta^{(i)})$, where $\Theta^{(i)} \in R^{s_{i+1} \times (s_i + 1) }$ is the matrix of weights mapping values from Layer $i$ to Layer $i+1$

For a neural network of L layers, $h_\Theta(x) = A^{(L)} = g(A^{(L-1)T}\Theta^{(L-1)})$



## Terms Appendix

T := Task that needs to be solved

E := Experience of many attempts to solve the task

P := The probability that the task will be solved

$m$ := the number of training examples

$h$ := our hypothesis $\ni h: \textbf{x} \rightarrow \textbf{y}$

$\alpha$ := the "learning" rate or step-size of the optimization step
