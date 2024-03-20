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

    @testset "Random" begin
        svd_trunc = TruncBondThresh(20, 0.0)
        L = 5
        q = (2, 4)
        d = 3
        C = rand_periodic_tt(d, L, q...)
        x = [[rand(1:q[1]), rand(1:q[2])] for _ in C]
        e1 = evaluate(C, x)

        orthogonalize_right!(C; svd_trunc)
        e2 = evaluate(C, x)
        @test e2 ≈ e1

        orthogonalize_left!(C; svd_trunc)
        e3 = evaluate(C, x)
        @test e3 ≈ e1
    end

    @testset "Uniform" begin
        svd_trunc = TruncBondThresh(20, 0.0)
        L = 5
        q = (2, 4)
        d = 3
        C = flat_periodic_tt(d, L, q...)
        x = [[rand(1:q[1]), rand(1:q[2])] for _ in C]
        e1 = evaluate(C, x)

        orthogonalize_right!(C; svd_trunc)
        e2 = evaluate(C, x)
        @test e2 ≈ e1

        orthogonalize_left!(C; svd_trunc)
        e3 = evaluate(C, x)
        @test e3 ≈ e1
    end

    @testset "Sampling" begin
        rng = MersenneTwister(0)
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                L = 6
                A = rand_periodic_tt( rand(rng, 2:7, L-1), qs... )
                x, p = sample(A)
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
                A = flat_periodic_tt( rand(2:7, L-1), qs... )
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
        A = flat_periodic_tt( rand(5:20, L-1), qs... )
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
        A = flat_periodic_tt( rand(5:20, L-1), qs... )
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
        A = flat_periodic_tt(rand(5:20, L-1), qs... )
        x1 = sample(MersenneTwister(1234), A)[1]
        orthogonalize_left!(A; svd_trunc)
        x2 = sample(MersenneTwister(1234), A)[1]
        @test x1 == x2
    end

    @testset "Exact" begin
        L = 3
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                A = rand_periodic_tt([1; rand(1:7, L-1); 1], qs... )
                m = marginals(A)
                m_exact = exact_marginals(A)
                @test m ≈ m_exact
                m2 = twovar_marginals(A)
                m2_exact = exact_twovar_marginals(A)
                @test m2 ≈ m2_exact
                @test exact_norm(A) ≈ norm(A)
            end
        end
    end

    @testset "Norm" begin
        L = 3; N = 4; q = 2; qs = fill(q, N)
        A = rand_periodic_tt( [1; rand(1:3, L-1); 1], qs... )
        B = rand_periodic_tt( [1; rand(1:3, L-1); 1], qs... )
        @test norm(A-B)^2 ≈ exact_norm(A-B)^2 ≈ norm2m(A,B)
    end

end
