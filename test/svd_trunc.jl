L = 10; N = 3; q = 10; qs = fill(q, N)
εs = 10.0 .^ (0.0:-0.5:-5)

@testset "TruncThresh" begin
    B = rand_tt( [1; rand(1:3, L-1); 1], qs... )
    normalize!(B)
    for ε in εs
        svd_trunc = TruncThresh(ε)
        A = deepcopy(B)
        compress!(A; svd_trunc)
        @test norm2m(A, B) < (L*ε)^2
    end
end

@testset "TruncBondThresh" begin
    B = rand_tt( [1; rand(1:5, L-1); 1], qs... )
    normalize!(B)
    for ε in εs
        svd_trunc = TruncBondThresh(2, ε)
        A = deepcopy(B)
        compress!(A; svd_trunc)
        @test norm2m(A, B) < (L*ε)^2
    end
end

@testset "TruncInfinite" begin
    p = rand_infinite_uniform_tt(10, 2, 4)
    q = deepcopy(p)
    compress!(p; svd_trunc=TruncInfinite(12))
    @test p == q
    compress!(p; svd_trunc=TruncInfinite(9))
    @test isapprox(real(only(marginals(p))), real(only(marginals(q))), atol=1e-5)
end

@testset "Merge and split tensors" begin
    A = rand(2,5,3)
    B = rand(5,4,2)
    C = TensorTrains._merge_tensors(A, B)
    C_test = map(Iterators.product(1:2,1:4,1:3,1:2)) do (i,j,xA,xB)
        dot(A[i,:,xA], B[:,j,xB])
    end
    @test C ≈ C_test
    x = [3,2]
    A_, B_ = TensorTrains._split_tensor(C; lr=TensorTrains.Left())
    @test A[:,:,x[1]] * B[:,:,x[2]] ≈ A_[:,:,x[1]] * B_[:,:,x[2]]
    A_, B_ = TensorTrains._split_tensor(C; lr=TensorTrains.Right())
    @test A[:,:,x[1]] * B[:,:,x[2]] ≈ A_[:,:,x[1]] * B_[:,:,x[2]]
end