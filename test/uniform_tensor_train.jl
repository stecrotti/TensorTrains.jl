@testset "Uniform Tensor Trains" begin

    @testset "Concrete uniform TT" begin
        rng = MersenneTwister(1)
        tensor = rand(rng, 4,4,2,3)
        L = 5
        A = UniformTensorTrain(tensor, L)
        @test bond_dims(A) == fill(4, L)
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
        rng = MersenneTwister(1)
        tensors = [rand(1,3,2,2), rand(3,4,2,2), rand(4,10,2,2), rand(10,1,2,2)]
        C = TensorTrain(tensors)
        A = symmetrized_uniform_tensor_train(C)
        x = sample(rng, C)[1]
        y = sum(evaluate(C, circshift(x,i)) for i in eachindex(x))
        @test all(evaluate(A, circshift(x,i)) ≈ y for i in eachindex(x))
    end

end