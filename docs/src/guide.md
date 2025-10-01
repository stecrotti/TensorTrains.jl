## Types
```@docs
AbstractTensorTrain
AbstractPeriodicTensorTrain
TensorTrain
PeriodicTensorTrain
```

## Functions
```@docs
normalize_eachmatrix!
flat_tt
rand_tt
flat_periodic_tt
rand_periodic_tt
bond_dims
evaluate
nparams
is_in_domain
marginals
twovar_marginals
lognormalization
normalization
normalize!
+
-
dot
norm
norm2m
sample!
sample
orthogonalize_right!
orthogonalize_left!
orthogonalize_two_site_center!
is_two_site_canonical
compress!
```

## Uniform Tensor Trains
```@docs
AbstractUniformTensorTrain
UniformTensorTrain
InfiniteUniformTensorTrain
symmetrized_uniform_tensor_train
periodic_tensor_train
```

### Infinite Tensor Trains
```@docs
TruncVUMPS
```

## Truncators
```@docs
SVDTrunc
TruncThresh
TruncBond
TruncBondMax
TruncBondThresh
```

## Gradients
```@docs
grad_squareloss_two_site
```

## DMRG-like optimization
```@docs
two_site_dmrg!
```

## Matrix Product States
Can be accessed, for now, via
```julia
using TensorTrains.MatrixProductStates
```

```@docs
TensorTrains.MatrixProductStates.MPS
TensorTrains.MatrixProductStates.rand_mps
TensorTrains.MatrixProductStates.loglikelihood
TensorTrains.MatrixProductStates.grad_loglikelihood_two_site
TensorTrains.MatrixProductStates.grad_normalization_two_site_canonical
```