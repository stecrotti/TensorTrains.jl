using TensorTrains, TensorTrains.MatrixProductStates
using Random
using Plots

rng = Xoshiro(1)
d_original = 2
L = 5
F = ComplexF64
q = rand_mps(rng, F, d_original, L, 4)
normalize!(q)
nsamples = 10^4
X = [sample(rng, q)[1] for _ in 1:nsamples]
nll = -loglikelihood(q, X)
println("Negative Log-Likelihood according to generating distribution q = $nll\n")
mq = marginals(q)
pmq = twovar_marginals(q)

p = rand_mps(rng, F, 2, length(q), 4)

function CB()
    nlls = zeros(0)
    diff_marg = zeros(0)
    diff_pair_marg = zeros(0)
    dots = zeros(0)
    function cb(sweep, k, p, nll)
        p_cp = deepcopy(p)
        normalize!(p_cp)
        d = abs(dot(p_cp.ψ, q.ψ))
        mbd = maximum(bond_dims(p.ψ))
        mp = marginals(p)
        d_m = maximum(maximum.(abs, mp-mq))
        pmp = twovar_marginals(p)
        d_pm = maximum(maximum.(abs, pmp-pmq))
        println("# Sweep $sweep, site k=$k")
        println("Negative LogLikelihood = $nll.\ndmax=$mbd")
        println("|<p|q>|=$d")
        println("Max diff marginals = $d_m")
        println("Max diff pair marginals = $d_pm\n")
        push!(nlls, -loglikelihood(p, X))
        push!(diff_marg, d_m)
        push!(diff_pair_marg, d_pm)
        push!(dots, d)
    end
end

callback = CB()
nsweeps = 50
two_site_dmrg!(p, X, nsweeps;
    η=1e-4, ndesc=10, svd_trunc=TruncBond(d_original), callback)

pl1 = plot(callback.nlls, xlabel="it", ylabel="NLL", label="")
hline!(pl1, [nll], ls=:dash, c=:gray, label="NLL according to generative model")
# vline!(pl1, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
pl2 = plot(callback.diff_marg, xlabel="it", ylabel="Max error on marginals", label="")
# vline!(pl2, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
pl2b = plot(callback.diff_pair_marg, xlabel="it", ylabel="Max error on pair marginals", label="")
pl3 = plot(callback.dots, xlabel="it", ylabel="<p|q>", label="")
# vline!(pl3, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
plot(pl1, pl2, pl2b, pl3, layout=(4,1), size=(500,900), margin=5Plots.mm)