using BenchmarkTools
using TensorTrains
import TensorTrains: accumulate_L, accumulate_R
using TensorTrains.MatrixProductStates

SUITE = BenchmarkGroup()

L = 10
qs = (2, 5, 3)
d = 10
q = rand_tt(d, L, qs...)
p = rand_periodic_tt(d, L+1, qs...)

SUITE["accumulators"] = BenchmarkGroup()
SUITE["accumulators"]["accumL_tensortrain"] = @benchmarkable accumulate_L($q)
# SUITE["accumulators"]["accumR_tensortrain"] = @benchmarkable accumulate_R($q)
SUITE["accumulators"]["accumL_periodic"] = @benchmarkable accumulate_L($p)
# SUITE["accumulators"]["accumR_periodic"] = @benchmarkable accumulate_R($p)

SUITE["marginals"] = BenchmarkGroup()
SUITE["marginals"]["marginals_tensortrain"] = @benchmarkable marginals($q)
SUITE["marginals"]["marginals_periodic"] = @benchmarkable marginals($p)

SUITE["twovar_marginals"] = BenchmarkGroup()
SUITE["marginals"]["twovar_marginals_tensortrain"] = @benchmarkable twovar_marginals($q)
SUITE["marginals"]["twovar_marginals_periodic"] = @benchmarkable twovar_marginals($p)

SUITE["orthogonalize"] = BenchmarkGroup()
svd_trunc = TruncThresh(0.0)
SUITE["orthogonalize"]["orth_left_tensortrain"] = @benchmarkable orthogonalize_left!($q; svd_trunc=$svd_trunc)
SUITE["orthogonalize"]["orth_left_periodic"] = @benchmarkable orthogonalize_left!($p; svd_trunc=$svd_trunc)

SUITE["sampling"] = BenchmarkGroup()
x = [[rand(1:q) for q in qs] for _ in 1:L]
function nsamples!(x, q, n)
    for _ in 1:n
        sample!(x, q)
    end
end
SUITE["sampling"]["sample_tensortrain"] = @benchmarkable nsamples!($x, $q, 20)
SUITE["sampling"]["sample_periodic"] = @benchmarkable nsamples!($x, $p, 20)

SUITE["dot"] = BenchmarkGroup()
q2 = rand_tt(d, L, qs...)
SUITE["dot"]["dot_tensortrain"] = @benchmarkable dot($q, $q2)
p2 = rand_periodic_tt(d, L+1, qs...)
SUITE["dot"]["dot_periodic"] = @benchmarkable dot($p, $p2)

SUITE["mps"] = BenchmarkGroup()
L = 5
qs = (2,)
d = 4
q = rand_mps(ComplexF64, d, L, qs...)
nsamples = 10^3
X = [sample(q)[1] for _ in 1:nsamples]
p = rand_mps(ComplexF64, d, L, qs...)
nsweeps = 1

SUITE["mps"]["dmrg"] = @benchmarkable begin
    two_site_dmrg!(p_cp, X, nsweeps; 
        η=5e-2, ndesc=100, svd_trunc=TruncBond(d))
end setup = (p_cp = deepcopy(p))
