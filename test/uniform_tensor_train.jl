@testset "Uniform Tensor Trains" begin
    rng = MersenneTwister(1)
    tensor = rand(rng, 4,4,2,3)
    L = 5
    A = UniformTensorTrain(tensor, L)
    A.z = rand(rng)

    @testset "Basics" begin
        @test bond_dims(A) == fill(4, L)
        B = UniformTensorTrain(tensor, L)
        @test A == B
        C = UniformTensorTrain(tensor .+ 1e-16, L)
        @test C ≈ A
    end

    @testset "Normalization" begin
        AA = deepcopy(A)
        normalize!(AA)
        @test float(normalization(AA)) ≈ 1
    end

    @testset "Concrete uniform TT" begin
        B = periodic_tensor_train(A)
        x = sample(rng, B)[1]
        @test evaluate(A, x) == evaluate(B, x)
        @test evaluate(A + A, x) ≈ 2 * evaluate(A, x)

        @test normalization(A) ≈ normalization(B)
        @test norm(A) ≈ norm(B)
        tensor2 = rand(rng, 3,3,2,3)
        C = UniformTensorTrain(tensor2, L)
        D = PeriodicTensorTrain(fill(tensor2, L)) 
        @test dot(A, C) == dot(B, D)
        @test dot(A, D) == dot(B, C)
        
        mA = marginals(A)
        mB = marginals(B)
        @test all(mb ≈ ma for (ma,mb) in zip(mA,mB))

        pA = twovar_marginals(A)
        pB = twovar_marginals(B)
        @test all(pb ≈ pa for (pa, pb) in zip(pA, pB))

        xA = sample(MersenneTwister(1), A)
        xB = sample(MersenneTwister(1), B)
        @test xA == xB
    end

    @testset "Symmetrized uniform TT" begin
        alltypes = [rand_tt([1,4,3,1], 2, 3), rand_periodic_tt([3,1,4], 1, 2), A]
        for B in alltypes
            S = symmetrized_uniform_tensor_train(B)
            x = sample(rng, B)[1]
            y = sum(evaluate(B, circshift(x,i)) for i in eachindex(x))
            @test all(evaluate(S, circshift(x,i)) ≈ y for i in eachindex(x))
        end
    end

    @testset "Errors" begin
        @test_throws ArgumentError (A[3] = rand(rng, 4,4,2,3))
        @test_throws "Not implemented" orthogonalize_left!(A)
        @test_throws "Not implemented" orthogonalize_right!(A)
        tensor = rand(rng, 4,4,2,3)
        B = UniformTensorTrain(tensor, L)
        @test_throws "Not implemented" A - B
    end
end

@testset "Infinite Uniform Tensor Train" begin
    rng = MersenneTwister(1)
    tensor = rand(rng, 4,4,2,3)
    A = InfiniteUniformTensorTrain(tensor)
    A.z = 3.5
    C = rand_infinite_uniform_tt(3, q)
    B = UniformTensorTrain(tensor, 100)
    D = UniformTensorTrain(C.tensor, 100)

    @testset "Base" begin
        @test A[1] === A.tensor
        @test length(A) == 1
    end

    @testset "Normalization" begin
        B = deepcopy(A)
        normalize!(B)
        normalize_eachmatrix!(B)
        @test float(normalization(B)) ≈ 1
        T = 20
        C = UniformTensorTrain(tensor, T; z = (A.z)^T)
        @test isapprox(normalization(A)^T, float(normalization(C)))
    end

    @testset "Marginals" begin
        @test marginals(A) ≈ marginals(B)
    end

    @testset "Plus" begin
        @test isapprox(marginals(A+A), marginals(B+B); atol=1e-6)
    end

    @testset "Marginals" begin
        marg = only(marginals(A))
        @test_throws DomainError twovar_marginals(A; maxdist=-2)
        tv = twovar_marginals(A; maxdist=3)
        @test tv[1,2] == tv[2,3] == tv[3,4]
        @test tv[1,3] == tv[2,4]
        two_marg = tv[1,2]
        N = ndims(two_marg)
        N2 = N ÷ 2
        @test sum(two_marg, dims=N2+1:N)[:,:,1,1] ≈ sum(two_marg, dims=1:N2)[1,1,:,:] ≈ marg
    end
end

@testset "Transfer operator" begin
    rng = MersenneTwister(0)
    L = 6
    A = rand(rng, 2,2,3,4)
    M = rand(rng, 3,3,3,4)
    q = UniformTensorTrain(A, L)
    q.z = 2.1
    p = UniformTensorTrain(M, L)
    p.z = 0.5

    G = transfer_operator(q, p)

    λ, l, r = leading_eig(transfer_operator(q, p))
    @test l * G ≈ l * λ
    @test G * r ≈ λ * r

    r = flat_infinite_uniform_tt(2, 3, 4)
    @test dot(r, r) ≈ 1
end

@testset "VUMPS truncations" begin
    rng = MersenneTwister(0)
    A = rand(rng, 10,10,3,4)
    p = InfiniteUniformTensorTrain(A)
    q = deepcopy(p)
    compress!(p; svd_trunc=TruncVUMPS(8))
    @test size(p.tensor)[1:2] == (8, 8)
    marg = real(only(marginals(q)))
    marg_compressed = real(only(marginals(p)))
    @test isapprox(marg, marg_compressed, atol=1e-5)
end