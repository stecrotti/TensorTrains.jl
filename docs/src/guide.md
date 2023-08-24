# Types

```@docs
AbstractTensorTrain
TensorTrain
PeriodicTensorTrain
```

# Functions
```@docs
normalize_eachmatrix!
uniform_tt
rand_tt
uniform_periodic_tt
rand_periodic_tt
bond_dims
evaluate
marginals
twovar_marginals
normalization
normalize!(::TensorTrain)
+
-
norm
sample!
sample
orthogonalize_right!
orthogonalize_left!
compress!
```

# SVD Truncators
```@docs
SVDTrunc
TruncThresh
TruncBond
TruncBondMax
TruncBondThresh
```