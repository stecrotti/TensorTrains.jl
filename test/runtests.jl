using TensorTrains
using Random, Suppressor, InvertedIndices
using Test

include("exact.jl")

svd_trunc = TruncThresh(0.0)
@suppress begin
    @show svd_trunc
end

@testset "single variable" begin
    tensors = [rand(1,3,2), rand(3,4,2), rand(4,10,2), rand(10,1,2)]
    C = TensorTrain(tensors)
    x = [rand(1:2,1) for _ in C]
    e1 = evaluate(C, x)

    orthogonalize_right!(C; svd_trunc)
    e2 = evaluate(C, x)
    @test e2 ≈ e1

    orthogonalize_left!(C; svd_trunc)
    e3 = evaluate(C, x)
    @test e3 ≈ e1

    e4 = evaluate(C, only.(x))
    @test e4 ≈ e1
end

@testset "two variables" begin
    tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
    C = TensorTrain(tensors)
    x = [rand(1:2,2) for _ in C]
    e1 = evaluate(C, x)

    orthogonalize_right!(C; svd_trunc)
    e2 = evaluate(C, x)
    @test e2 ≈ e1

    orthogonalize_left!(C; svd_trunc)
    e3 = evaluate(C, x)
    @test e3 ≈ e1
end

@testset "random" begin
    L = 5
    q = (2, 4)
    d = 3
    C = rand_tt(d, L, q...)
    x = [[rand(1:q[1]), rand(1:q[2])] for _ in C]
    e1 = evaluate(C, x)

    orthogonalize_right!(C; svd_trunc)
    e2 = evaluate(C, x)
    @test e2 ≈ e1

    orthogonalize_left!(C; svd_trunc)
    e3 = evaluate(C, x)
    @test e3 ≈ e1
end

@testset "Accumulators" begin
    tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
    A = TensorTrain(tensors)
    L = TensorTrains.accumulate_L(A)
    R = TensorTrains.accumulate_R(A)
    @test L[end] ≈ R[begin]
end

@testset "Sum of TTs" begin
    for N in 1:3
        for q in 1:3
            qs = fill(q, N)
            L = 6
            A = rand_tt( [1; rand(1:7, L-1); 1], qs... )
            B = rand_tt( [1; rand(1:7, L-1); 1], qs... )
            x = [rand(1:q[1],N) for _ in 1:L]
            @test evaluate(A, x) + evaluate(B, x) ≈ evaluate(A+B, x)
        end
    end
end

@testset "Difference of TTs" begin
    rng = MersenneTwister(0)
    for N in 1:3
        for q in 1:3
            qs = fill(q, N)
            L = 6
            A = rand_tt( [1; rand(1:7, L-1); 1], qs... )
            B = rand_tt( [1; rand(1:7, L-1); 1], qs... )
            x = [rand(1:q[1],N) for _ in 1:L]
            @test evaluate(A, x) - evaluate(B, x) ≈ evaluate(A - B, x)
        end
    end
end

@testset "Sample TT" begin
    rng = MersenneTwister(0)
    for N in 1:3
        for q in 1:3
            qs = fill(q, N)
            L = 6
            A = rand_tt( [1; rand(rng, 1:7, L-1); 1], qs... )
            x, p = sample(rng, A)
            normalize!(A)
            @test evaluate(A, x) ≈ p
        end
    end
end

@testset "Exact" begin
    L = 3
    for N in 1:3
        for qs in 1:3
            A = rand_tt( [1; rand(1:7, L-1); 1], qs... )
            m = marginals(A)
            m_exact = exact_marginals(A)
            @test m ≈ m_exact
        end
    end
end