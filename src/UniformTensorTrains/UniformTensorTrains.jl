module UniformTensorTrains

using ..TensorTrains
using ..TensorTrains: _reshape1

using LinearAlgebra
using KrylovKit: eigsolve
using Tullio: @tullio

export AbstractPeriodicTensorTrain, PeriodicTensorTrain, flat_periodic_tt, rand_periodic_tt,
       AbstractUniformTensorTrain, UniformTensorTrain, periodic_tensor_train,
       symmetrized_uniform_tensor_train, InfiniteUniformTensorTrain,
       transfer_operator, infinite_transfer_operator, leading_eig


include("uniform_tensor_train.jl")
include("transfer_operator.jl")


end # module