"""
    TruncVUMPS{TI, TF} <: SVDTrunc

A type used to perform  truncations of an [`InfiniteUniformTensorTrain`](@ref) to a target bond size `d`.
It uses the Variational Uniform Matrix Product States (VUMPS) algorithm from MPSKit.jl.

# FIELDS
- `d`: target bond dimension
- `maxiter = 100`: max number of iterations for the VUMPS algorithm
- `tol = 1e-14`: tolerance for the VUMPS algorithm

```@example
p = rand_infinite_uniform_tt(10, 2, 2)
compress!(p, TruncVUMPS(5))
```
"""
struct TruncVUMPS{TI<:Integer, TF<:Real} <: SVDTrunc
    d       :: TI
    maxiter :: TI
    tol     :: TF
end
TruncVUMPS(d::Integer; maxiter=100, tol=1e-14) = TruncVUMPS(d, maxiter, tol)

summary(svd_trunc::TruncVUMPS) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)

function truncate_vumps(A::Array{F,3}, d; 
        init = rand(d, size(A,2), d),
        maxiter = 100, kw_vumps...) where {F}
    ψ = InfiniteMPS([TensorMap(init, (ℝ^d ⊗ ℝ^size(A,2)), ℝ^d)])
    Q = size(A, 2)
    m = size(A, 1)
    @assert size(A, 3) == m
    t = TensorMap(A,(ℝ^m ⊗ ℝ^Q), ℝ^m) # the same but as a type digestible by MPSKit.jl
    ψ₀ = InfiniteMPS([t])
    II = DenseMPO([add_util_leg(id(storagetype(site_type(ψ₀)), physicalspace(ψ₀, i)))
        for i in 1:length(ψ₀)])
    alg = VUMPS(; maxiter, verbosity=0, kw_vumps...) # variational approximation algorithm
    # alg = IDMRG1(; maxiter)
    @assert typeof(ψ) == typeof(ψ₀)
    ψ_, = approximate(ψ, (II, ψ₀), alg)   # do the truncation
    @assert typeof(ψ) == typeof(ψ_)

    ovl = abs(dot(ψ_, ψ₀))
    B = reshape(only(ψ_.AL).data, d, Q, d)
    return B, ovl, ψ_
end

function compress!(A::InfiniteUniformTensorTrain; svd_trunc::TruncVUMPS=TruncVUMPS(4),
        is_orthogonal::Symbol=:none, init=rand_infinite_uniform_tt(svd_trunc.d, size(A.tensor)[3:end]...))
    (; d, maxiter, tol) = svd_trunc
    qs = size(A.tensor)[3:end]
    B = reshape(A.tensor, size(A.tensor)[1:2]..., prod(qs))
    Bperm = permutedims(B, (1,3,2))
    Btruncperm, = truncate_vumps(Bperm, d; maxiter, tol, init = permutedims(_reshape1(init.tensor), (1,3,2)))
    Btrunc = permutedims(Btruncperm, (1,3,2))
    A.tensor = reshape(Btrunc, size(Btrunc)[1:2]..., qs...)
    return A
end