using TensorTrains, Optim
import UniformIsingModels
using Random, LinearAlgebra, Unzip, Statistics
using Plots

N = 20
J = 1.0
rng = MersenneTwister(0)
h = randn(rng, N)
β = 1.0
ising = UniformIsingModels.UniformIsing(N, J, h, β)
T = 10^3
spin2int(σ) = 2 - (σ==-1)
int2spin(x) = 2x-3
X = [[rand(1:2) for _ in 1:N] for _ in 1:T]
Y = [UniformIsingModels.energy(ising, int2spin.(x)) for x in X]

Ttest = 10^2
Xtest = [[rand(1:2) for _ in 1:N] for _ in 1:Ttest]
Ytest = [UniformIsingModels.energy(ising, int2spin.(x)) for x in Xtest]

ψ = rand_tt(2, N, 2)

function CB()
    loss = zeros(0)
    dmax = zeros(0)
    loss_test = zeros(0)
    function cb(sweep, k, it, ψ, ll)
        if it == 1
            preds = [evaluate(ψ, x) for x in X]
            l = mean(abs2, preds - Y)
            push!(loss, l)
            preds_test = [evaluate(ψ, x) for x in Xtest]
            l_test = mean(abs2, preds_test - Ytest)
            push!(loss_test, l_test)
            mbd = maximum(bond_dims(ψ))
            push!(dmax, mbd)
            println("# site k=$k")
            println("Loss=$l\ndmax=$mbd")
        end
    end
end

callback = CB()
nsweeps = 6
ndesc = 100
η = 1e-3
svd_trunc=TruncBond(8)

two_site_dmrg!(ψ, X, Y, nsweeps; η, ndesc, svd_trunc, callback,
    optimizer = Optim.Adam(alpha=η), weight_decay=1e-3)

pl = plot(; xlabel="Iterations", title="Uniform Ising with N=$N, $T training data", titlefontsize=11)
plot!(pl, callback.loss, label="Training loss")
plot!(pl, callback.loss_test, label="Test loss")
vline!(pl, (N-1):(N-1):length(callback.loss), label="Ends of sweeps", ls=:dash, c=:gray)
