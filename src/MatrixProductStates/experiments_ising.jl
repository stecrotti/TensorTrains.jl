# Fitting a simple ising model with fully connected couplings

using TensorTrains, TensorTrains.MatrixProductStates
import UniformIsingModels
using Random, LinearAlgebra, Unzip, Statistics
using Plots

N = 100
J = 1.0
rng = MersenneTwister(0)
h = randn(rng, N)
β = 1.0
ising = UniformIsingModels.UniformIsing(N, J, h, β)
nsamples = 5*10^3
spin2int(σ) = 2 - (σ==-1)
int2spin(x) = 2x-3
S, ps = unzip([UniformIsingModels.sample(ising) for _ in 1:nsamples])
X = [[spin2int(σi) for σi in σ] for σ in S]
means_data = mean(X)
ll_data = mean(log, ps)
println("Log-Likelihood according to generating distribution = $ll_data\n")

p = MPS(rand_tt(ComplexF64, 2, N, 2))

function CB()
    lls = zeros(0)
    ds = zeros(0)
    dmax = zeros(0)
    function cb(sweep, k, it, p, ll)
        if it == 1
            means_p = [dot(eachindex(m), m) for m in marginals(p)]
            d_m = mean(abs, means_data - means_p)
            mbd = maximum(bond_dims(p.ψ))
            println("# site k=$k")
            println("LogLikelihood=$ll\ndmax=$mbd")
            println("Mean diff empirical vs fitted means = $d_m\n")
            push!(lls, ll)
            push!(ds, d_m)
            push!(dmax, mbd)
        end
    end
end

callback = CB()
nsweeps = 3
ndesc = 1
η = 5e-2
svd_trunc=TruncBond(10)

two_site_dmrg!(p, X, nsweeps; η, ndesc, svd_trunc, callback)

println("Log-Likelihood according to generating distribution = $ll_data\n")

pl = plot(; xlabel="Iterations", title="Uniform Ising with N=$N, $nsamples training data", titlefontsize=11)
plot!(pl, callback.lls, label="Log-Likelihood")
hline!(pl, [ll_data], ls=:dot, c=:black, label="Log-Likelihood according to generating distr")
vline!(pl, (N-1):(N-1):length(callback.lls), label="Ends of sweeps", ls=:dash, c=:gray)
pl2 = plot(callback.ds; yscale=:log10, label="Mean difference in empirical vs fitted means", xlabel="Iterations")
vline!(pl2, (N-1):(N-1):length(callback.lls), label="Ends of sweeps", ls=:dash, c=:gray, legend=:topright)
pl3 = plot(callback.dmax, label="Max bond dim", xlabel="Iterations", legend=:bottomright)
vline!(pl3, (N-1):(N-1):length(callback.lls), label="Ends of sweeps", ls=:dash, c=:gray)
plot(pl, pl2, pl3, layout=(3,1), size=(500,500))