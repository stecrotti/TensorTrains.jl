L = 10; N = 3; q = 10; qs = fill(q, N)
εs = 10.0 .^ (0.0:-0.5:-5)

@testset "TruncThresh" begin
    B = rand_tt( [1; rand(1:3, L-1); 1], qs... )
    normalize!(B)
    for ε in εs
        svd_trunc = TruncThresh(ε)
        A = deepcopy(B)
        compress!(A; svd_trunc)
        @test normAminusB(A, B) < L*ε
    end
end

@testset "TruncBondThresh" begin
    B = rand_tt( [1; rand(1:5, L-1); 1], qs... )
    normalize!(B)
    for ε in εs
        svd_trunc = TruncBondThresh(2, ε)
        A = deepcopy(B)
        compress!(A; svd_trunc)
        @test normAminusB(A, B) < L*ε
    end
end