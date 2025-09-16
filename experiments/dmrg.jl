using TensorTrains, TensorTrains.MatrixProductStates

d_original = 3
L = 5
q = rand_mps(ComplexF64, d_original, L, 2,2)
normalize!(q)
nsamples = 5*10^3
X = [sample(q)[1] for _ in 1:nsamples]
nll = -loglikelihood(q, X)
println("Negative Log-Likelihood according to generating distribution q=$nll\n")
mq = marginals(q)

# Use complex entries for (supposed) better expressivity
p = MPS(rand_tt(ComplexF64, 2, length(q), 2,2))

function CB()
    function cb(sweep, k, p, nll)
        p_cp = deepcopy(p)
        normalize!(p_cp)
        d = abs(dot(p_cp.ψ,q.ψ))
        mbd = maximum(bond_dims(p.ψ))
        mp = marginals(p)
        d_m = maximum(maximum.(abs, mp-mq))
        println("# Sweep $sweep, site k=$k")
        println("Negative LogLikelihood=$nll.\n\tdmax=$mbd")
        println("|<p|q>|=$d")
        println("Max diff marginals = $d_m")
    end
end

callback = CB()
nsweeps = 4
two_site_dmrg!(p, X, nsweeps;
    η=1e-3, ndesc=10, svd_trunc=TruncBond(d_original), callback)
