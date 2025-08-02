# Fitting a simple ising model with fully connected couplings

using TensorTrains, TensorTrains.MatrixProductStates
import UniformIsingModels
using Random, LinearAlgebra, Unzip, Statistics

N = 10
J = 1.0
rng = MersenneTwister(0)
h = randn(rng, N)
β = 1.0
ising = UniformIsingModels.UniformIsing(N, J, h, β)
nsamples = 5*10^3
spin2int(σ) = 2 - (σ==-1)
int2spin(x) = 2x-3
S, ps = unzip([UniformIsingModels.sample(ising) for _ in 1:nsamples])
X = [[[spin2int(σi)] for σi in σ] for σ in S]
means_data = only.(mean(X))
ll_data = mean(log, ps)
println("Log-Likelihood according to generating distribution = $ll_data\n")

p = MPS(rand_tt(ComplexF64, 2, N, 2))

function CB()
    function cb(it, p, k, ll)
        if it == 1
            means_p = [dot(eachindex(m), m) for m in marginals(p)]
            d_m = maximum(abs, means_data - means_p)
            mbd = maximum(bond_dims(p.ψ))
            println("# site k=$k")
            println("it=$it.\nLogLikelihood=$ll\ndmax=$mbd")
            println("Max diff empirical vs fitted means = $d_m\n")
        end
    end
end

callback = CB()
nsweeps = 2
two_site_dmrg!(p, X, nsweeps; 
    η=5e-2, ndesc=10, svd_trunc=TruncBond(2), callback)