module MatrixProductStates

using TensorTrains
import TensorTrains: _reshape1, accumulate_L, accumulate_R, sample_noalloc, 
    normalize!, _merge_tensors
using Lazy: @forward
using Tullio: @tullio
using Random: AbstractRNG, default_rng
using StatsBase
using LinearAlgebra: I, tr
using LogarithmicNumbers: Logarithmic

export MPS
export is_left_canonical, is_right_canonical, is_canonical
export grad_normalization_canonical, grad_normalization_two_site_canonical,
    loglikelihood, grad_loglikelihood

include("mps.jl")
include("derivatives.jl")
include("dmrg.jl")

end # module