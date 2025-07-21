using TensorTrains.MatrixProductStates
import TensorTrains.MatrixProductStates: trace
import TensorTrains: accumulate_L, accumulate_R
using StatsBase: sample
using Tullio: @tullio
using LinearAlgebra: I

@testset "Matrix Product States" begin

    rng = MersenneTwister(0)

    @testset "No-trace tensor train" begin
        tensors = [rand(ComplexF64, 1,5,2,2), rand(ComplexF64, 5,4,2,2),
            rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,1,2,2)]
        ψ = TensorTrain(tensors)
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

        @testset "Normalization with z≠0" begin
            p = MPS(deepcopy(ψ))
            p.ψ.z = 2
            q = exact_prob(p)
            z = float(normalization(p))
            @test z ≈ sum(q)
            @test z ≈ abs2(norm(p.ψ))
            p_cp = deepcopy(p)
            x, px = sample(MersenneTwister(0), p_cp)
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
end