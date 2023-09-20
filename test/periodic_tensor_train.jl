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
        @test float(normalization(A)) ≈ float(normalization(B))
        @test evaluate(A + A, x) ≈ evaluate(B + B, x)
    end

    @testset "Accumulators" begin
        tensors = [rand(4,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,4,2,2)]
        A = PeriodicTensorTrain(tensors)
        l, = TensorTrains.accumulate_L(A; normalize=false)
        r, = TensorTrains.accumulate_R(A; normalize=false)
        m = TensorTrains.accumulate_M(A)
        Z = float(normalization(A))
        @test Z ≈ exact_normalization(A)
        @test TensorTrains.accumulate_R(A)[2] ≈ Z
        @test tr(l[end]) ≈ Z
        @test tr(r[begin]) ≈ Z
        @test tr(l[begin] * m[1,end] * r[end]) ≈ Z
    end

    @testset "Compression" begin
        tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
        A = PeriodicTensorTrain(tensors)
        x, = sample(A)
        B = deepcopy(A)
        C = deepcopy(A)
        svd_trunc = TruncThresh(1e-2)
        compress!(A; svd_trunc)
        orthogonalize_right!(B; svd_trunc = TruncThresh(0.0))
        compress!(B; svd_trunc, is_orthogonal=:right)
        orthogonalize_left!(C; svd_trunc = TruncThresh(0.0))
        compress!(C; svd_trunc, is_orthogonal=:left)
        @test evaluate(A, x) ≈ evaluate(B, x) ≈ evaluate(C, x)
        @test_throws ArgumentError compress!(A; is_orthogonal=:something)
    end


    @testset "Long tensor trains" begin
        rng = MersenneTwister(0)
        qs = (2, 2)
        L = 10000

        # overflow
        A = rand_periodic_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        l, = TensorTrains.accumulate_L(A; normalize=false)
        @test any(isinf, only(l[end]))
        @test !isinf(lognormalization(A))
        normalize!(A)
        @test float(normalization(A)) ≈ 1

        compress!(A, svd_trunc=TruncThresh(0.0))
        @test !isinf(normalization(A))

        # underflow
        A = rand_periodic_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        A.tensors .*= 1e-50
        l, = TensorTrains.accumulate_L(A; normalize=false)
        @test any(iszero, only(l[end]))
        @test !iszero(lognormalization(A))
        normalize!(A)
        @test float(normalization(A)) ≈ 1

        compress!(A, svd_trunc=TruncThresh(0.0))
        @test !isinf(normalization(A))
    end

    @testset "Negative values" begin
        rng = MersenneTwister(0)
        qs = (2, 3)
        L = 4
        A = rand_periodic_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        A[1] .*= -1
        Z = exact_normalization(A)
        @test float(normalization(A)) ≈ Z
        @test Z < 0

        @testset "Attempt sampling from tensor train with negative values" begin
            @test_throws ErrorException sample(A)
        end
    end


    @testset "Normalize eachmatrix" begin
        rng = MersenneTwister(0)
        qs = (2, 4)
        L = 4
        A = rand_periodic_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        A.z = -100 * rand(rng)
        x, p = sample(rng, A)
        e = evaluate(A, x)
        z = normalization(A)
        normalize_eachmatrix!(A)
        @test evaluate(A, x) ≈ e
        @test float(normalization(A)) ≈ float(z)
    end

    @testset "Flat" begin
        L = 5
        bondsizes = rand(1:4, L)
        q = (2,4,3)
        C = flat_periodic_tt(bondsizes, q...)
        @assert float(normalization(C)) ≈ 1
    end

    @testset "Random" begin
        svd_trunc = TruncBondThresh(20, 0.0)
        L = 5
        q = (2, 4)
        d = 3
        C = rand_periodic_tt(d, L, q...)
        x = [[rand(1:q[1]), rand(1:q[2])] for _ in C]
        e1 = evaluate(C, x)
        z1 = float(normalization(C))

        orthogonalize_right!(C; svd_trunc)
        e2 = evaluate(C, x)
        z2 = float(normalization(C))
        @test e2 ≈ e1
        @test z2 ≈ z1

        orthogonalize_left!(C; svd_trunc)
        e3 = evaluate(C, x)
        z3 = float(normalization(C))
        @test e3 ≈ e1
        @test z3 ≈ z1
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
                A = rand_periodic_tt(rand(1:7, L), qs...)
                m = marginals(A)
                m_exact = exact_marginals(A)
                @test m ≈ m_exact
                m2 = twovar_marginals(A)
                m2_exact = exact_twovar_marginals(A)
                @test m2 ≈ m2_exact
                @test exact_norm(A) ≈ norm(A)
                B = rand_periodic_tt(rand(1:7, L), qs...)
                @test exact_dot(A, B) ≈ dot(A, B)
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
