using TensorTrains.MatrixProductStates
import TensorTrains.MatrixProductStates: trace
import TensorTrains: accumulate_L, accumulate_R
using StatsBase: sample
using Tullio: @tullio
using LinearAlgebra: I

@testset "Matrix Product States" begin

    rng = MersenneTwister(0)

    tensors = [rand(ComplexF64, 1,5,2,2), rand(ComplexF64, 5,4,2,2),
        rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,1,2,2)]
    ψ = TensorTrain(tensors)
    p = MPS(ψ)
    q = exact_prob(p)
    z = float(normalization(p))

    @testset "Checks against exact computations" begin
        @test exact_normalization(p) ≈ z
        @test exact_marginals(p) ≈ marginals(p)
    end

    @testset "Normalization" begin
        @test z ≈ sum(q)
        @test z ≈ abs2(norm(p.ψ))
        L, zL = accumulate_L(p; normalize=false)
        R, zR = accumulate_R(p; normalize=false)
        @test float(zL) ≈ float(zR)
        @test trace(L[end]) ≈ trace(R[begin])
        @test z ≈ trace(L[end])
        p_cp = deepcopy(p)
        x, px = sample(MersenneTwister(0), p_cp)
        logz_cp = normalize!(p_cp)
        @test float(normalization(p_cp)) ≈ 1
        @test exp(logz_cp) ≈ z
        x_, px_ = sample(MersenneTwister(0), p_cp)
        @test x == x_
        @test float(px) ≈ float(px_)
    end

    @testset "Normalization with z≠0" begin
        p = MPS(deepcopy(ψ))
        p.ψ.z = 2
        q = exact_prob(p)
        z = float(normalization(p))
        @test z ≈ sum(q)
        @test z ≈ abs2(norm(p.ψ))
        p_cp = deepcopy(p)
        x, px = sample(MersenneTwister(0), p_cp)
        logz_cp = normalize!(p_cp)
        @test float(normalization(p_cp)) ≈ 1
        @test exp(logz_cp) ≈ z
        normalize!(p_cp)
        @test float(normalization(p_cp)) ≈ 1
        x_, px_ = sample(MersenneTwister(0), p_cp)
        @test x == x_
        @test float(px) ≈ float(px_)
    end

    @testset "Orthogonalization" begin
        x = [[rand(rng, 1:q) for q in size(A)[3:end]] for A in ψ]
        px = evaluate(p, x)
        orthogonalize_left!(p; svd_trunc=TruncThresh(0))
        @test evaluate(p, x) ≈ px
        L = accumulate_L(p; normalize=false)[1]
        @test all(reshape(Lˡ,size(Lˡ)[2:2:4]...) ≈ Matrix(I, size(Lˡ)[2:2:4]...) for Lˡ in L[1:end-1])
        orthogonalize_right!(p; svd_trunc=TruncThresh(0))
        @test evaluate(p, x) ≈ px
        R = accumulate_R(p; normalize=false)[1]
        @test all(reshape(Rˡ,size(Rˡ)[1:2:3]...) ≈ Matrix(I, size(Rˡ)[1:2:3]...) for Rˡ in R[2:end])
        compress!(p; svd_trunc=TruncThresh(0))
        @test evaluate(p, x) ≈ px
    end

    @testset "Orthogonalization and normalization" begin
        tensors = [rand(ComplexF64, 1,5,2,2), rand(ComplexF64, 5,4,2,2),
            rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,1,2,2)]
        ψ = TensorTrain(tensors)
        ψ.z = 2
        p = MPS(ψ)
        z = float(normalization(p))
        for l in eachindex(ψ)
            orthogonalize_center!(ψ, l)
            Aˡ = ψ[l]
            @tullio zz = conj(Aˡ[m,n,x1,x2]) * Aˡ[m,n,x1,x2]
            @test float(zz / abs2(ψ.z)) ≈ z
        end
    end

    @testset "Sampling" begin
        x, q = sample(rng, p)
        normalize!(p)
        @test q ≈ evaluate(p, x)
    end
end

@testset "Trace tensor train" begin
    tensors = [rand(ComplexF64, 3,5,2,2), rand(ComplexF64, 5,4,2,2),
        rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,3,2,2)]
    ψ = PeriodicTensorTrain(tensors)
    p = MPS(ψ)
    q = exact_prob(p)
    z = float(normalization(p))

    @testset "Normalization" begin
        @test z ≈ sum(q)
        @test z ≈ abs2(norm(p.ψ))
        L, zL = accumulate_L(p; normalize=false)
        R, zR = accumulate_R(p; normalize=false)
        @test float(zL) ≈ float(zR)
        @test trace(L[end]) ≈ trace(R[begin])
        @test z ≈ trace(L[end])
        p_cp = deepcopy(p)
        x, px = sample(MersenneTwister(0), p_cp)
        normalize!(p_cp)
        @test float(normalization(p_cp)) ≈ 1
        x_, px_ = sample(MersenneTwister(0), p_cp)
        @test x == x_
        @test float(px) ≈ float(px_)
    end

    @testset "Sampling" begin
        x, q = sample(rng, p)
        normalize!(p)
        @test q ≈ evaluate(p, x)
    end
end

# Test against finite differences
@testset "Real Derivatives" begin
    F = Float64
    tensors = [rand(F, 1,5,2,2), rand(F, 5,4,2,2),
        rand(F, 4,10,2,2), rand(F, 10,1,2,2)]
    ψ = TensorTrain(tensors)
    p = MPS(ψ)
    normalize!(p)

    function compute_dzdA_numeric(p, l; ε = 1e-8 * one(eltype(p[l])))
        A = p[l]
        maxi, maxj, maxxi, maxxj = size(A)

        return map(Iterators.product(1:maxi, 1:maxj, 1:maxxi, 1:maxxj)) do (i, j, xi, xj)
            function f(a)
                p_cp = deepcopy(p)
                p_cp[l][i,j,xi,xj] = a
                return normalization(p_cp; normalize_while_accumulating=false)
            end
            a = A[i,j,xi,xj]
            float((f(a+ε) - f(a)) / ε)
        end
    end

    @testset "Gradient of Z" begin
        for l in eachindex(p)
            orthogonalize_center!(p, l)
            dzdA, z = grad_normalization_canonical(p, l)
            @test z ≈ normalization(p)
            ε = 1e-8 * one(eltype(p[l]))
            dzdA_numeric = compute_dzdA_numeric(p, l; ε)
            @test all(abs.(dzdA - dzdA_numeric) .< 10ε)
        end
    end

    # @testset "Gradient of Z - 2 site" begin
    #     for l in 1:length(p)-1  
    #         orthogonalize_two_site_center!(p, l)
    #         @assert is_two_site_canonical(p, l)
    #         dzdA, z = grad_normalization_two_site_canonical(p, l)
    #         @test z ≈ float(normalization(p))
    #         ε = 1e-8 * one(eltype(p[l]))
    #         dzdA_numeric = compute_two_site_dzdA_numeric(p, l; ε)
    #         d = abs.(dzdA - dzdA_numeric)
    #         @test all(d .< 10ε)
    #     end
    # end

    @testset "Gradient of log of unnormalized prob" begin
        X = sample(p)[1]

        for l in eachindex(X)
            orthogonalize_center!(p, l)
            A = p[l]
            x = X[l]
            ε = 1e-8 * one(eltype(A))
            maxi, maxj, _ = size(A)

            gr, val = grad_evaluate(p.ψ, l, X)
            dlldA = 2 * gr / val

            dlldA_numeric = map(Iterators.product(1:maxi, 1:maxj)) do (i, j)
                function f(a)
                    p_cp = deepcopy(p)
                    @assert is_canonical(p, l)
                    p_cp[l][i,j, x...] = a
                    return log(evaluate(p_cp, X))
                end

                a = A[i,j,x...]
                float((f(a+ε) - f(a)) / ε)
            end

            d = dlldA - dlldA_numeric
            @test all(abs.(d) .< 100ε)
        end
    end

    @testset "Gradient of loglikelihood" begin
        X = [sample(p)[1] for _ in 1:10]
        l = 2
        orthogonalize_center!(p, l)
        dlldA, ll = grad_loglikelihood(p, l, X)
        @test ll ≈ loglikelihood(p, X)
        A = p[l]
        ε = 1e-8 * one(eltype(A))
        maxi, maxj, maxxi, maxxj = size(A)

        dlldA_numeric = map(Iterators.product(1:maxi, 1:maxj, 1:maxxi, 1:maxxj)) do (i, j, xi, xj)
            function f(a)
                p_cp = deepcopy(p)
                @assert is_canonical(p, l)
                p_cp[l][i,j,xi,xj] = a
                return loglikelihood(p_cp, X)
            end

            a = A[i,j,xi,xj]
            float((f(a+ε) - f(a)) / ε)
        end
        d = dlldA - dlldA_numeric
        
        @test all(abs.(d) .< 100ε)
    end

    @testset "Gradient of loglikelihood - 2-site" begin
        X = [sample(p)[1] for _ in 1:10]
        for l in 1:length(p)-1
            orthogonalize_two_site_center!(p, l)
            p_cp = deepcopy(p)
            A = _merge_tensors(p_cp[l], p_cp[l+1])
            dlldA, ll = grad_loglikelihood_two_site(p_cp, l, X)
            @test ll ≈ loglikelihood(p_cp, X)
            η = 1e-3
            lls = map(1:100) do _
                A = _merge_tensors(p_cp[l], p_cp[l+1])
                dlldA, ll = grad_loglikelihood_two_site(p_cp, l, X)
                @test ll ≈ loglikelihood(p_cp, X)
                p_cp[l], p_cp[l+1] = TensorTrains._split_tensor(A + η*dlldA)
                ll
            end
            @test issorted(lls) # check that log-likelihood is increasing steadily for small learning rate
        end
    end

# @testset "DMRG" begin
#     q = MPS(rand_tt(3, 5, 2,2))
#     normalize!(q)
#     X = [sample(q)[1] for _ in 1:10^4]
#     @show loglikelihood(q, X)
#     mq = marginals(q)
#     p = MPS(rand_tt(2, length(q), 2,2))

#     function CB()
#         function cb(it, p, k, ll)
#             p_cp = deepcopy(p)
#             normalize!(p_cp)
#             if it == 1
#                 d = float(norm2m(p_cp.ψ,q.ψ))
#                 mbd = maximum(bond_dims(p.ψ))
#                 mp = marginals(p)
#                 d_m = maximum(maximum.(abs, mp-mq))
#                 println("# k=$k")
#                 println("\tit=$it. LogLikelihood=$ll. dmax=$mbd")
#                 println("\t|p-q|^2=$d")
#                 println("\tMax diff marginals $d_m")
#             end
#         end
#     end
#     callback = CB()
#     nsweeps = 3
#     two_site_dmrg!(p, X, nsweeps; η=5e-2, svd_trunc=TruncBond(3), callback)

#     mp = marginals(p)
# end
end