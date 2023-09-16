@testset "Uniform Tensor Trains" begin
    rng = MersenneTwister(1)
    tensor = rand(rng, 4,4,2,3)
    L = 5
    A = UniformTensorTrain(tensor, L)

    @testset "Basics" begin
        @test bond_dims(A) == fill(4, L)
        B = UniformTensorTrain(tensor, L)
        @test A == B
        C = UniformTensorTrain(tensor .+ 1e-16, L)
        @test C ≈ A
    end

    @testset "Concrete uniform TT" begin
        B = periodic_tensor_train(A)
        x = sample(rng, B)[1]
        @test evaluate(A,x) == evaluate(B, x)
        @test evaluate(A + A,x) ≈ 2 * evaluate(A, x)

        @test normalization(A) == normalization(B)
        @test norm(A) == norm(B)
        tensor2 = rand(rng, 3,3,2,3)
        C = UniformTensorTrain(tensor2, L)
        D = PeriodicTensorTrain(fill(tensor2, L)) 
        @test dot(A, C) == dot(B, D)
        @test dot(A, D) == dot(B, C)
        
        mA = marginals(A)
        mB = marginals(B)
        @test all(mb ≈ only(mA) for mb in mB)

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
        @test_throws ArgumentError (A[3] .= 1)
        @test_throws "Not implemented" orthogonalize_left!(A)
        @test_throws "Not implemented" orthogonalize_right!(A)
        @test_throws "Not implemented" compress!(A)
        tensor = rand(rng, 4,4,2,3)
        B = UniformTensorTrain(tensor, L)
        @test_throws "Not implemented" A - B
    end

end