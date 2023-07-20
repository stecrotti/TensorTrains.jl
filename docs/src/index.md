## What is a Tensor Train?
A [Tensor Train](https://tensornetwork.org/mps/) is a type of tensor factorization involving the product of 3-index tensors organized on a one-dimensional chain. 
In the context of function approximation and probability, a function of $L$ discrete variables is in Tensor Train format if it is written as
```math
f(x^1, x^2, \ldots, x^L) = \sum_{a^1,a^2,\ldots,a^{L-1}} [A^1(x^1)]_{a^1}[A^2(x^2)]_{a^1,a^2}\cdots [A^{L-1}(x^{L-1})]_{a^{L-2},a^{L-1}}[A^L(x^L)]_{a^{L-1}}
```
where, for every choice of $x^l$, $A^l(x^l)$ is a real-valued matrix and the matrix sizes must be compatible.
The first matrix must have 1 row and the last matrix should have 1 column, such that the product correctly returns a scalar.

The Tensor Train factorization can be used to parametrize probability distributions, which is the main focus of this package. In this case, $f$ should be properly normalized and always return a non-negative value. 

### Tensor Trains with Periodic Boundary Conditions
A slight generalization, useful to describe systems with periodic boundary conditions is the following:
```math
f(x^1, x^2, \ldots, x^L) = \sum_{a^1,a^2,\ldots,a^{L}} [A^1(x^1)]_{a^1,a^2}[A^2(x^2)]_{a^2,a^3}\cdots [A^{L-1}(x^{L-1})]_{a^{L-1},a^{L}}[A^L(x^L)]_{a^{L},a^1}
```
In other words, to evaluate $f$ one takes the trace of the product of matrices.

## Notation and terminology
Tensor Trains are the most basic type of [Tensor Network](https://tensornetwork.org/). Tensor networks are a large family of tensor factorizations which are often best represented in diagrammatic notation. For this reason, the term _bond_ is used interchangeably as _index_. The indices $a^1,a^2,\ldots,a^{L-1}$ are usually called the _virtual indices_, while $x^1, x^2, \ldots, x^L$ are the _physical indices_.

Tensor Trains are used to parametrize wavefunctions in many-body quantum physics. The resulting quantum state is called [Matrix Product State](https://en.wikipedia.org/wiki/Matrix_product_state). In such context, the entries are generally complex numbers, and a probability can be obtained for a given state by taking the squared absolute value of the wavefunction.

In this package we focus on the "classical" case where the Tensor Train directly represents a probability distribution $p(x^1, x^2, \ldots, x^L)$. 

## Efficient computation
Given a Tensor Train some simple recursive strategies can be employed to

### Compute the normalization
```math
Z = \sum_{x^1, x^2, \ldots, x^L} \sum_{a^1,a^2,\ldots,a^{L-1}} [A^1(x^1)]_{a^1}[A^2(x^2)]_{a^1,a^2}\cdots [A^{L-1}(x^{L-1})]_{a^{L-2},a^{L-1}}[A^L(x^L)]_{a^{L-1}}
```
such that 
```math
\begin{aligned}
1&=\sum_{x^1, x^2, \ldots, x^L}p(x^1, x^2, \ldots, x^L)\\&=\sum_{x^1, x^2, \ldots, x^L}\frac1Z \sum_{a^1,a^2,\ldots,a^{L-1}} [A^1(x^1)]_{a^1}[A^2(x^2)]_{a^1,a^2}\cdots [A^{L-1}(x^{L-1})]_{a^{L-2},a^{L-1}}[A^L(x^L)]_{a^{L-1}}
\end{aligned}
```
### Compute marginals
Single-variable
```math
p(x^l=x) = \sum_{x^1, x^2, \ldots, x^L} p(x^1, x^2, \ldots, x^L) \delta(x^l,x)
```
and two-variable
```math
p(x^l=x, x^m=x') = \sum_{x^1, x^2, \ldots, x^L} p(x^1, x^2, \ldots, x^L) \delta(x^l,x)\delta(x^m,x')
```
### Extract exact samples
Via hierarchical sampling
```math
p(x^1, x^2, \ldots, x^L) = p(x^1)p(x^2|x^1)p(x^3|x^1,x^2)\cdots p(x^L|x^1,x^2,\ldots,x^{L-1})
```
by first sampling $x^1\sim p(x^1)$, then $x^2\sim p(x^2|x^1)$ and so on.

## What can this package do?
This small package provides some utilities for creating, manipulating and evaluating Tensor Trains interpreted as functions, with a focus on the probabilistic side. 
Each variable $x^l$ is assumed to be multivariate.
Whenever performing some probability-related operation, it is responsability of the user to make sure that the Tensor Train always represents a valid probability distribution.

Common operations are:

- `evaluate` a Tensor Train at a given set of indices
- `orthogonalize_left!`, `orthogonalize_right!`: bring a Tensor Train to [left/right orthogonal form](https://tensornetwork.org/mps/)
- `compress!` a Tensor Train using [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)-based truncations
- `normalize!` a Tensor Train in the probability sense (not in the $L_2$ norm sense!), see above
- `sample` from a Tensor Train intended as a probability ditribution, see above
- `+`,`-`: take the sum/difference of two TensorTrains

### Example
Let's construct and initialize at random a Tensor Train of the form
```math
f\left((x^1,y^1), (x^2,y^2), (x^3,y^3)\right) = \sum_{a^1,a^2} [A^1(x^1,y^1)]_{a^1}[A^2(x^2,y^2)]_{a^1,a^2}[A^3(x^3,y^3)]_{a^2}
```
where $x^l\in\{1,2\}, y^l\in\{1,2,3\}$.
```julia
using TensorTrains
L = 3        # length
q = (2, 3)   # number of values taken by x, y
d = 5        # bond dimension
A = rand_tt(d, L, q...)    # construct Tensor Train with random positive entries
xy = [[rand(1:qi) for qi in q] for _ in 1:L]    # random set of indices
p = evaluate(A, xy)    # evaluate `A` at `xy`
compress!(A; svd_trunc=TruncThresh(1e-8));    # compress `A` to reduce the bond dimension
pnew = evaluate(A, xy)
ε = abs((p-pnew)/p)
```

## References
- https://tensornetwork.org: "an open-source review article focused on tensor network algorithms, applications, and software"
- Oseledets, I.V., 2011. [Tensor-train decomposition](https://sites.pitt.edu/~sjh95/related_papers/tensor_train_decomposition.pdf). SIAM Journal on Scientific Computing, 33(5).