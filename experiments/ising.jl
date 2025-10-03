# Fitting a simple ising model with fully connected couplings

using TensorTrains, TensorTrains.MatrixProductStates, Optim
import UniformIsingModels
using Random, LinearAlgebra, Unzip, Statistics
using Plots

N = 20
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
nll_data = -mean(log, ps)
println("Negative Log-Likelihood according to generating distribution = $nll_data\n")
C_true = UniformIsingModels.covariances(ising)
C_data = cov(X)

function marginal_to_covariance(M)
    avg_σᵢσⱼ = avg_σᵢ = avg_σⱼ = 0.0
    for xⱼ in 1:2
        for xᵢ in 1:2
            p = M[xᵢ,xⱼ]
            avg_σᵢσⱼ += p * xᵢ * xⱼ
            avg_σᵢ += p * xᵢ
            avg_σⱼ += p * xⱼ
        end
    end
    return avg_σᵢσⱼ - avg_σᵢ * avg_σⱼ
end

function compute_covariances(p)
    pm = twovar_marginals(p)
    N = length(p)
    C_mps = zeros(N, N)
    for j in 1:N
        for i in 1:j-1
            C_mps[i,j] =  C_mps[j,i] = marginal_to_covariance(pm[i,j])
        end
    end
    return C_mps
end

function CB()
    nlls = zeros(0)
    ds = zeros(0)
    dmax = zeros(0)
    ds_cov = zeros(0)
    function cb(sweep, k, p, nll)
        means_p = [dot(eachindex(m), m) for m in marginals(p)]
        d_m = mean(abs, means_data - means_p)
        mbd = maximum(bond_dims(p.ψ))
        C_mps = compute_covariances(p)
        diffs = C_mps - C_data
        d_c = mean(abs, diffs[i,j] for i in 1:N for j in i+1:N)
        println("# Sweep $sweep, site k=$k")
        println("Negative LogLikelihood=$nll\ndmax=$mbd")
        println("Mean diff empirical vs fitted means = $d_m\n")
        println("Mean diff empirical vs fitted covariance = $d_c\n")
        push!(nlls, nll)
        push!(ds, d_m)
        push!(dmax, mbd)
        push!(ds_cov, d_c)
    end
end


nsweeps = 5
ndesc = 100
η = 1e-4

ds = 5:5:50
plots = []
nlls = []
diff_mean = []
diff_cov = []
times = []

for (a, d) in enumerate(ds)
    println("##### d=$d #####")
    p = rand_mps(Float64, 2, N, 2)
    callback = CB()
    t = @timed two_site_dmrg!(p, X, nsweeps; η, ndesc, svd_trunc=TruncBond(d), callback)
    println("Negative Log-Likelihood according to generating distribution = $nll_data\n")
    pl = plot(; xlabel="Iterations", title="Uniform Ising with N=$N, $nsamples training data, d=$d", titlefontsize=11)
    plot!(pl, callback.nlls; yscale=:log10, label="Negative Log-Likelihood")
    hline!(pl, [nll_data], ls=:dot, c=:black, label="Negative Log-Likelihood according to generating distr")
    vline!(pl, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
    pl2 = plot(callback.ds; yscale=:log10, label="Mean difference in empirical vs fitted means", xlabel="Iterations")
    vline!(pl2, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray, legend=:topright)
    # pl3 = plot(callback.dmax, label="Max bond dim", xlabel="Iterations", legend=:bottomright)
    # vline!(pl3, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
    pl4 = plot(callback.ds_cov; yscale=:log10, label="Mean difference in empirical vs fitted covariances", xlabel="Iterations", legend=:topright)
    vline!(pl4, (N-1):(N-1):length(callback.nlls), label="Ends of sweeps", ls=:dash, c=:gray)
    pl_ising = plot(pl, pl2, pl4, layout=(3,1), size=(500,800))
    push!(plots, pl_ising)
    push!(nlls, last(callback.nlls))
    push!(diff_mean, last(callback.ds))
    push!(diff_cov, last(callback.ds_cov))
    push!(times, t.time)
    println("\n\n")
end

pl_precision_cov = plot(ds, diff_cov, xlabel="d", ylabel="Mean abs error on covariances", m=:o,
    yscale=:log10, label="")
plot!(pl_precision_cov, title="Learning Ising N=$N from $nsamples samples")
savefig(pl_precision_cov, "experiments/ising_bonddims_N$(N).pdf")

# savefig(pl_ising, "experiments/ising_train.pdf")

# pl_ising