using BenchmarkTools
using TensorTrains

SUITE = BenchmarkGroup()

L = 10
qs = (2, 5, 3)
d = 10
q = rand_tt(d, L, qs...)

SUITE["marginals"] = BenchmarkGroup()
SUITE["marginals"]["marginals_tensortrain"] = @benchmarkable marginals($q)

SUITE["twovar_marginals"] = BenchmarkGroup()
SUITE["marginals"]["twovar_marginals_tensortrain"] = @benchmarkable twovar_marginals($q)

SUITE["orthogonalize"] = BenchmarkGroup()
SUITE["orthogonalize"]["orth_left_tensortrain"] = @benchmarkable orthogonalize_left!($q)

SUITE["sampling"] = BenchmarkGroup()
x = [[rand(1:q) for q in qs] for _ in 1:L]
function nsamples!(x, q, n)
    for _ in 1:n
        sample!(x, q)
    end
end
SUITE["sampling"]["sample_tensortrain"] = @benchmarkable nsamples!($x, $q, 20)

SUITE["dot"] = BenchmarkGroup()
q2 = rand_tt(d, L, qs...)
SUITE["dot"]["dot_tensortrain"] = @benchmarkable dot($q, $q2)
