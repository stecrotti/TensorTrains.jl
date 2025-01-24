using OffsetArrays

svd_trunc = TruncThresh(0.0)
@suppress begin
    @show svd_trunc
end

rng = MersenneTwister(0)

@testset "TensorTrain" begin

    @testset "basics" begin
        tensors = [rand(1,5,2,2), rand(5,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
        A = TensorTrain(tensors)
        B = TensorTrain(copy(tensors))
        @test A == B
        @test A ≈ B
    end

    @testset "Bond dimensions" begin
        tensors = [rand(1,4,2,2), rand(3,5,2,2)]
        @test TensorTrains.check_bond_dims(tensors) == false
    end

    @testset "Offsets" begin
        f(i,j) = rand(i,j,5)
        tensors = [f(1,5), f(5,6), f(6,2), f(2,1)]
        A = TensorTrain(tensors)
        tensors2 = [OffsetArray(t, axes(t,1), axes(t,2), -2:2) for t in tensors]
        B = TensorTrain(tensors2)
        @test normalization(A) == normalization(B)
        x = rand(-2:2, 4)
        C = deepcopy(B)
        compress!(B)
        @test evaluate(C,x) ≈ evaluate(B,x)
    end

    @testset "Complex" begin
        f(i,j) = rand(ComplexF64, i, j, 5)
        tensors = [f(1,5), f(5,6), f(6,2), f(2,1)]
        A = TensorTrain(tensors)
        B = deepcopy(A)
        compress!(A)
        x = rand(1:5, 4)
        @test evaluate(A,x) ≈ evaluate(B,x)
    end

    @testset "single variable" begin
        tensors = [rand(1,3,2), rand(3,4,2), rand(4,10,2), rand(10,1,2)]
        C = TensorTrain(tensors)
        C.z = 12.3
        z1 = float(normalization(C))

        @test bond_dims(C) == [1,3,4,10]
        @test eltype(C) == eltype(1.0)

        x = [rand(1:2,1) for _ in C]
        e1 = evaluate(C, x)

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

        e4 = evaluate(C, only.(x))
        @test e4 ≈ e1
    end

    @testset "two variables" begin
        svd_trunc = TruncBond(20)

        tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
        C = TensorTrain(tensors)
        x = [rand(1:2,2) for _ in C]
        e1 = evaluate(C, x)

        orthogonalize_right!(C; svd_trunc)
        e2 = evaluate(C, x)
        @test e2 ≈ e1

        svd_trunc = TruncBondMax(20)

        orthogonalize_left!(C; svd_trunc)
        e3 = evaluate(C, x)
        @test e3 ≈ e1

        compress!(C; svd_trunc=TruncThresh(1e-15))
        e4 = evaluate(C, x)
        @test e4 ≈ e1
    end

    @testset "Random" begin
        svd_trunc = TruncBondThresh(20, 0.0)

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

    @testset "Flat" begin
        q = (2, 4)
        bondsizes = [1, 3, 5, 2, 1]
        C = flat_tt(bondsizes, q...)
        @test float(normalization(C)) ≈ 1
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
        l, = TensorTrains.accumulate_L(A; normalize=false)
        r, = TensorTrains.accumulate_R(A; normalize=false)
        m = TensorTrains.accumulate_M(A)
        Z = float(normalization(A))
        @test Z ≈ exact_normalization(A)
        @test TensorTrains.accumulate_R(A)[2] ≈ Z
        @test only(l[end]) ≈ Z
        @test only(r[begin]) ≈ Z
        @test only(l[begin] * m[1,end] * r[end]) ≈ Z
    end

    @testset "Compression" begin
        tensors = [rand(rng, 1,3,2,2), rand(rng, 3,4,2,2), rand(rng, 4,10,2,2), 
            rand(rng, 10,1,2,2)]
        A = TensorTrain(tensors)
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
        qs = (2, 2)
        L = 10000

        # overflow
        A = rand_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        l, = TensorTrains.accumulate_L(A; normalize=false)
        @test any(isinf, only(l[end]))
        @test !isinf(lognormalization(A))
        normalize!(A)
        @test float(normalization(A)) ≈ 1

        compress!(A, svd_trunc=TruncThresh(0.0))
        @test !isinf(normalization(A))

        # underflow
        A = rand_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
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
        qs = (2, 4)
        L = 4
        A = rand_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
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
        A = rand_tt( [1; rand(rng, 10:15, L-1); 1], qs... )
        A.z = -100 * rand(rng)
        x, p = sample(rng, A)
        e = evaluate(A, x)
        z = normalization(A)
        normalize_eachmatrix!(A)
        @test evaluate(A, x) ≈ e
        @test float(normalization(A)) ≈ float(z)
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
        L = 4
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                A = rand_tt( [1; rand(1:7, L-1); 1], qs... )
                B = rand_tt( [1; rand(1:7, L-1); 1], qs... )
                x = [rand(1:q[1],N) for _ in 1:L]
                @test evaluate(A, x) - evaluate(B, x) ≈ evaluate(A - B, x)
            end
        end
    end

    @testset "Sampling" begin
        rng = MersenneTwister(0)
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                L = 6
                A = rand_tt( [1; rand(rng, 1:7, L-1); 1], qs... )
                x, p = sample(copy(rng), A)
                y = deepcopy(x)
                sample!(rng, y, A)
                @test x == y
                normalize!(A)
                @test evaluate(A, x) ≈ p
                x, p = sample!(x, A)
                @test evaluate(A, x) ≈ p
            end
        end
    end

    @testset "Marginals" begin
        L = 3
        for N in 1:3
            for q in 1:3
                qs = fill(q, N)
                A = rand_tt([1; rand(1:7, L-1); 1], qs... )
                m = marginals(A)
                m_exact = exact_marginals(A)
                @test m ≈ m_exact
                m2 = twovar_marginals(A)
                m2_exact = exact_twovar_marginals(A)
                @test m2 ≈ m2_exact
            end
        end
    end

    @testset "Norm" begin
        L = 3; N = 2; q = 2; qs = fill(q, N)
        A = rand_tt( [1; rand(1:3, L-1); 1], qs... )
        B = rand_tt( [1; rand(1:3, L-1); 1], qs... )
        @test norm(A-B)^2 ≈ exact_norm(A-B)^2 ≈ norm2m(A,B)
    end
end
