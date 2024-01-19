using TensorTrains.MatrixProductStates
import TensorTrains: accumulate_L, accumulate_R
using StatsBase: sample

tensors = [rand(ComplexF64, 1,5,2,2), rand(ComplexF64, 5,4,2,2),
    rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,1,2,2)]
ψ = TensorTrain(tensors)
p = MatrixProductState(ψ)

L = accumulate_L(p)
R = accumulate_R(p)
@test only(L[end]) ≈ only(R[begin])
@test normalization(p) ≈ abs2(norm(p.ψ))

rng = MersenneTwister(0)
x, q = sample(rng, p)
normalize!(p)
@test q ≈ evaluate(p, x)