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

@testset "TruncVUMPS" begin
    p = flat_infinite_uniform_tt(10, 2, 4)
    q = deepcopy(p)
    compress!(p; svd_trunc=TruncVUMPS(12))
    @test p == q
    compress!(p; svd_trunc=TruncVUMPS(9))
    @test isapprox(real(only(marginals(p))), real(only(marginals(q))), atol=1e-5)
end