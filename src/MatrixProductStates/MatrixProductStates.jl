module MatrixProductStates

using TensorTrains
import TensorTrains: _reshape1, accumulate_L, accumulate_R, sample_noalloc, 
    normalize!, _merge_tensors, _split_tensor, LeftOrRight, Left, Right,
    precompute_left_environments, precompute_right_environments
using Lazy: @forward
using Tullio: @tullio
using Random: AbstractRNG, default_rng
using StatsBase
using LinearAlgebra: I, tr
using LogarithmicNumbers: Logarithmic
import Optim

export MPS
export rand_mps
export is_left_canonical, is_right_canonical, is_canonical
export grad_normalization_canonical, grad_normalization_two_site_canonical,
    loglikelihood, grad_loglikelihood, grad_loglikelihood_two_site,
    two_site_dmrg_sweep!, two_site_dmrg!

include("mps.jl")
include("derivatives.jl")
include("dmrg.jl")

end # module