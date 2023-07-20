@testset "PeriodicTensorTrain" begin

    @testset "TensorTrain as a subcase" begin
        tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
        A = TensorTrain(tensors)
        @test PeriodicTensorTrain(tensors) == PeriodicTensorTrain(A)
        tensorsB = [cat(tensors[1], zeros(2,3,2,2), dims=1), 
                        tensors[2], tensors[3], 
                        cat(tensors[end], zeros(10,2,2,2), dims=2)
                    ]
        B = PeriodicTensorTrain(tensorsB)
        x = [rand(1:2,2) for _ in A]
        @test evaluate(A, x) ≈ evaluate(B, x)
        @test marginals(A) ≈ marginals(B)
        @test twovar_marginals(A) ≈ twovar_marginals(B)
        @test normalization(A) ≈ normalization(B)
        @test evaluate(A + A, x) ≈ evaluate(B + B, x)
    end

    @testset "Sample" begin
        rng = MersenneTwister(0)
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                L = 6
                A = rand_periodic_tt( rand(rng, 2:7, L-1), qs... )
                x, p = sample(rng, A)
                normalize!(A)
                @test evaluate(A, x) ≈ p
            end
        end
    end

    @testset "Sum of TTs" begin
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                L = 6
                A = uniform_periodic_tt( rand(2:7, L-1), qs... )
                B = rand_periodic_tt( rand(2:7, L-1), qs... )
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
                A = rand_periodic_tt( rand(2:7, L-1), qs... )
                B = rand_periodic_tt( rand(2:7, L-1), qs... )
                x = [rand(1:q[1],N) for _ in 1:L]
                @test evaluate(A, x) - evaluate(B, x) ≈ evaluate(A - B, x)
            end
        end
    end

    @testset "Orthogonalize right" begin
        svd_trunc = TruncThresh(3e-2)
        q = 4; N = 3; L = 6
        qs = fill(q, N)
        A = uniform_periodic_tt( rand(5:20, L-1), qs... )
        bd1 = bond_dims(A)
        x = [rand(MersenneTwister(1234), 1:q[1],N) for _ in 1:L]
        e1 = evaluate(A, x)
        orthogonalize_right!(A; svd_trunc)
        @assert sum(bond_dims(A)) < sum(bd1)
        e2 = evaluate(A, x)
        @test e1 ≈ e2
    end

    @testset "Orthogonalize left" begin
        svd_trunc = TruncThresh(3e-2)
        q = 4; N = 3; L = 6
        qs = fill(q, N)
        A = uniform_periodic_tt( rand(5:20, L-1), qs... )
        bd1 = bond_dims(A)
        x = [rand(MersenneTwister(1234), 1:q[1],N) for _ in 1:L]
        e1 = evaluate(A, x)
        orthogonalize_left!(A; svd_trunc)
        @assert sum(bond_dims(A)) < sum(bd1)
        e2 = evaluate(A, x)
        @test e1 ≈ e2
    end

    @testset "Orthogonalize + sampling" begin
        svd_trunc = TruncThresh(0.0)
        q = 4; N = 3; L = 6
        qs = fill(q, N)
        A = uniform_periodic_tt( rand(5:20, L-1), qs... )
        x1 = sample(MersenneTwister(1234), A)[1]
        orthogonalize_left!(A; svd_trunc)
        x2 = sample(MersenneTwister(1234), A)[1]
        @test x1 == x2
    end

end

nothing
