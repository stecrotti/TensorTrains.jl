using TensorTrains.MatrixProductStates
import TensorTrains.MatrixProductStates: trace
import TensorTrains: accumulate_L, accumulate_R
using StatsBase: sample


@testset "Matrix Product States" begin

    function exact_prob(p::MatrixProductState)
        qs = [size(Aˡ)[3:end] for Aˡ in p]
        X = Iterators.product((1:prod(qˡ) for (Aˡ,qˡ) in zip(p,qs))...)
        map(X) do x
            evaluate(p, [Tuple(CartesianIndices(qs[l])[x[l]]) for l in eachindex(x)])
        end
    end

    rng = MersenneTwister(0)

    @testset "No-trace tensor train" begin
        tensors = [rand(ComplexF64, 1,5,2,2), rand(ComplexF64, 5,4,2,2),
            rand(ComplexF64, 4,10,2,2), rand(ComplexF64, 10,1,2,2)]
        ψ = PeriodicTensorTrain(tensors)
        p = MatrixProductState(ψ)
        q = exact_prob(p)
        z = normalization(p)

        @testset "Normalization" begin
            L = accumulate_L(p)
            R = accumulate_R(p)
            @test trace(L[end]) ≈ trace(R[begin])
            @test z ≈ trace(L[end])
            @test z ≈ sum(q)
            @test z ≈ abs2(norm(p.ψ))
        end

        @testset "Orthogonalization" begin
            normalize!(p)
            x = [[rand(rng, 1:q) for q in size(A)[3:end]] for A in ψ]
            px = evaluate(p, x)
            orthogonalize_left!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
            L = accumulate_L(p)
            @test all(reshape(Lˡ,size(Lˡ)[2:2:4]...) ≈ Matrix(I, size(Lˡ)[2:2:4]...) for Lˡ in L)
            orthogonalize_right!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
            R = accumulate_R(p)
            @test all(reshape(Rˡ,size(Rˡ)[1:2:3]...) ≈ Matrix(I, size(Rˡ)[1:2:3]...) for Rˡ in R)
            compress!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
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
        p = MatrixProductState(ψ)
        q = exact_prob(p)

        @testset "Normalization" begin
            L = accumulate_L(p)
            R = accumulate_R(p)
            @test trace(L[end]) ≈ trace(R[begin])
            @test normalization(p) ≈ sum(q)
            @test normalization(p) ≈ abs2(norm(p.ψ))
        end

        @testset "Orthogonalization" begin
            x = [[rand(rng, 1:q) for q in size(A)[3:end]] for A in ψ]
            px = evaluate(p, x)
            orthogonalize_left!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
            L = accumulate_L(p)
            orthogonalize_right!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
            orthogonalize_right!(p; svd_trunc=TruncThresh(0))
            @test evaluate(p, x) ≈ px
        end

        @testset "Sampling" begin
            x, q = sample(rng, p)
            normalize!(p)
            @test q ≈ evaluate(p, x)
        end
    end
end

