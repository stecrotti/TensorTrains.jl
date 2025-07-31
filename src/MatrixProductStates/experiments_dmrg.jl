using TensorTrains, TensorTrains.MatrixProductStates

q = MPS(rand_tt(3, 5, 2,2))
normalize!(q)
nsamples = 5*10^3
X = [sample(q)[1] for _ in 1:nsamples]
ll = loglikelihood(q, X)
println("Log-Likelihood according to generating distribution q=$ll\n")
mq = marginals(q)
p = MPS(rand_tt(2, length(q), 2,2))

function CB()
    function cb(it, p, k, ll)
        if it == 1
            p_cp = deepcopy(p)
            normalize!(p_cp)
            d = float(norm2m(p_cp.ψ,q.ψ))
            mbd = maximum(bond_dims(p.ψ))
            mp = marginals(p)
            d_m = maximum(maximum.(abs, mp-mq))
            println("# site k=$k")
            println("\tit=$it. LogLikelihood=$ll. dmax=$mbd")
            println("\t|p-q|^2=$d")
            println("\tMax diff marginals $d_m")
        end
    end
end
callback = CB()
nsweeps = 2
two_site_dmrg!(p, X, nsweeps; η=5e-2, svd_trunc=TruncBond(3), callback)