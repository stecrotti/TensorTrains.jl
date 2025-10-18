using TensorTrains, TensorTrains.MatrixProductStates
using Plots

d_original = 3
L = 5
q = rand_mps(ComplexF64, d_original, L, 4)
normalize!(q)
nsamples = 5*10^3
X = [sample(q)[1] for _ in 1:nsamples]
nll = -loglikelihood(q, X)
println("Negative Log-Likelihood according to generating distribution q = $nll\n")
mq = marginals(q)

p = rand_mps(ComplexF64, 2, length(q), 4)

function CB()
    nlls = zeros(0)
    diff_marg = zeros(0)
    dots = zeros(0)
    function cb(sweep, k, p, nll)
        p_cp = deepcopy(p)
        normalize!(p_cp)
        d = abs(dot(p_cp.ψ,q.ψ))
        mbd = maximum(bond_dims(p.ψ))
        mp = marginals(p)
        d_m = maximum(maximum.(abs, mp-mq))
        println("# Sweep $sweep, site k=$k")
        println("Negative LogLikelihood = $nll.\ndmax=$mbd")
        println("|<p|q>|=$d")
        println("Max diff marginals = $d_m\n")
        push!(nlls, -loglikelihood(p, X))
        push!(diff_marg, d_m)
        push!(dots, d)
    end
end

callback = CB()
nsweeps = 20
two_site_dmrg!(p, X, nsweeps;
    η=1e-3, ndesc=10, svd_trunc=TruncBond(d_original), callback)

pl1 = plot(callback.nlls, xlabel="it", ylabel="NLL", label="")
hline!(pl1, [nll], ls=:dash, c=:gray, label="NLL according to generative model")
# vline!(pl1, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
pl2 = plot(callback.diff_marg, xlabel="it", ylabel="Max error on marginals", label="")
# vline!(pl2, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
pl3 = plot(callback.dots, xlabel="it", ylabel="<p|q>", label="")
# vline!(pl3, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
plot(pl1, pl2, pl3, layout=(3,1), size=(500,800), margin=5Plots.mm)