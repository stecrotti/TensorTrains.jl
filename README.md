# TensorTrains.jl

[![Build Status](https://github.com/stecrotti/TensorTrains.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stecrotti/TensorTrains.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/stecrotti/TensorTrains.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/stecrotti/TensorTrains.jl)

A [Tensor Train](https://tensornetwork.org/mps/), also known as Matrix Product State in physics, is a type of tensor factorization involving the product of 3-index tensors organized on a one-dimensional chain.
In the context of function approximation and probability, a function of $L$ discrete variables is in Tensor Train format if it is written as
```math
f(x^1, x^2, \ldots, x^L) = \sum_{a^1,a^2,\ldots,a^{L-1}} [A^1(x^1)]_{a^1}[A^2(x^2)]_{a^1,a^2}\cdots [A^{L-1}(x^{L-1})]_{a^{L-2},a^{L-1}}[A^L(x^L)]_{a^{L-1}}
```
where, for every choice of $x^l$, $A^l(x^l)$ is a real-valued matrix and the matrix sizes must be compatible.
In particular, the Tensor Train factorization can be used to parametrize probability distributions. In this case, $f$ should be properly normalized and always return a non-negative value. 

This package provides some utilities for creating, manipulating and evaluating Tensor Trains interpreted as functions, with a focus on the probabilistic side:

- `evaluate`
- `compress`
- `normalization`, `normalize!`
- `marginals`, `twovar_marginals`
- `sample`
