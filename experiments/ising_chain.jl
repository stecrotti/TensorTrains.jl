import IsingChains
using TensorTrains, TensorTrains.MatrixProductStates
using Random, Statistics, Unzip
using Plots

rng = Xoshiro(0)

N = 100
J = randn(rng, N-1)
h = randn(rng, N)
q = IsingChains.IsingChain(J, h, 1.0)
nsamples = 5*10^3
spin2int(σ) = 2 - (σ==-1)
S, ps = unzip([IsingChains.sample(q) for _ in 1:nsamples])
X = [[spin2int(σi) for σi in σ] for σ in S]
means_data = mean(X)
cov_data = cov(X)
nll_data = -mean(ps)
println("Negative Log-Likelihood according to generating distribution = $nll_data\n")

p = rand_mps(Float64, 2, N, 2)

function CB()
    nlls = zeros(0)
    ds = zeros(0)
    dmax = zeros(0)
    function cb(sweep, k, p, nll)
        means_p = [dot(eachindex(m), m) for m in marginals(p)]
        d_m = mean(abs, means_data - means_p)
        mbd = maximum(bond_dims(p.ψ))
        println("# Sweep $sweep, site k=$k")
        println("Negative LogLikelihood=$nll\ndmax=$mbd")
        println("Mean diff empirical vs fitted means = $d_m\n")
        push!(nlls, nll)
        push!(ds, d_m)
        push!(dmax, mbd)
    end
end

callback = CB()
nsweeps = 20
ndesc = 10
η = 5e-5
svd_trunc=TruncBond(4)

two_site_dmrg!(p, X, nsweeps; η, ndesc, svd_trunc, callback)

println("Negative Log-Likelihood according to generating distribution = $nll_data\n")

pl = plot(; xlabel="Iterations", title="Uniform Ising with N=$N, $nsamples training data", titlefontsize=11)
plot!(pl, callback.nlls, label="Negative Log-Likelihood")
hline!(pl, [nll_data], ls=:dot, c=:black, label="Negative Log-Likelihood according to generating distr")
pl2 = plot(callback.ds; yscale=:log10, label="Mean difference in empirical vs fitted means", xlabel="Iterations")
pl3 = plot(callback.dmax, label="Max bond dim", xlabel="Iterations", legend=:bottomright)
pl_ising = plot(pl, pl2, pl3, layout=(3,1), size=(500,500))

# savefig(pl_ising, "experiments/ising_chain_N$(N).pdf")
