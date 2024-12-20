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
compress!
```
### Uniform Tensor Trains
```@docs
AbstractUniformTensorTrain
UniformTensorTrain
InfiniteUniformTensorTrain
symmetrized_uniform_tensor_train
periodic_tensor_train
```

## Truncators
```@docs
SVDTrunc
TruncThresh
TruncBond
TruncBondMax
TruncBondThresh
```

### Infinite Tensor Trains
```@docs
TruncVUMPS
```