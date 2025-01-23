"""
    AbstractTensorTrain{F<:Number, N}

An abstract type representing a Tensor Train.
"""
abstract type AbstractTensorTrain{F<:Number, N} end

Base.eltype(::AbstractTensorTrain{F,N}) where {N,F} = F

"""
    AbstractPeriodicTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

An abstract type representing a Tensor Train with periodic boundary conditions.
"""
abstract type AbstractPeriodicTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N} end 


"""
    bond_dims(A::AbstractTensorTrain)

Return a vector with the dimensions of the virtual bonds
"""
bond_dims(A::AbstractTensorTrain) = [size(a, 1) for a in A]
###
function check_bond_dims(tensors::AbstractVector{T}) where {T<:AbstractArray}
    for t in 1:lastindex(tensors)
        firstindex(tensors[t],1) == firstindex(tensors[t],2) == 1 || 
            throw(ArgumentError("first two indices must start at 1"))
        dᵗ = size(tensors[t],2)
        dᵗ⁺¹ = size(tensors[mod1(t+1, length(tensors))],1)
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end

"""
    evaluate(A::AbstractTensorTrain, X...)

Evaluate the Tensor Train `A` at input `X`

Example:
```@example
    L = 3
    q = (2, 3)
    A = rand_tt(4, L, q...)
    X = [[rand(1:qi) for qi in q] for l in 1:L]
    evaluate(A, X)
```
"""
function evaluate(A::AbstractTensorTrain, X...)
    Ax = tr(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))
    return float(Ax / A.z)
end

Base.:(==)(A::T, B::T) where {T<:AbstractTensorTrain} = isequal(A.tensors, B.tensors)
Base.isapprox(A::T, B::T; kw...) where {T<:AbstractTensorTrain} = isapprox(A.tensors, B.tensors; kw...)

trace(At) = @tullio _[aᵗ,aᵗ⁺¹] := _reshape1(At)[aᵗ,aᵗ⁺¹,x]

function accumulate_L(A::AbstractTensorTrain{F}; normalize=true) where {F}
    Lt = Matrix(one(F)*I, size(A[begin],1), size(A[begin],1))
    z = Logarithmic(one(F))
    L = map(trace(Atx) for Atx in A) do At
        nt = maximum(abs, Lt)
        if !iszero(nt) && normalize
            Lt ./= nt
            z *= nt
        end
        Lt = Lt * At
    end
    z *= tr(Lt)
    return L, z
end

function accumulate_R(A::AbstractTensorTrain{F}; normalize=true) where {F}
    Rt = Matrix(one(F)*I, size(A[end],2), size(A[end],2))
    z = Logarithmic(one(F))
    R = map(trace(Atx) for Atx in Iterators.reverse(A)) do At
        nt = maximum(abs, Rt)
        if !iszero(nt) && normalize
            Rt ./= nt
            z *= nt
        end
        Rt = At * Rt
    end |> reverse
    z *= tr(Rt)
    return R, z
end

function accumulate_M(A::AbstractTensorTrain{F}) where {F}
    L = length(A)
    M = fill(zeros(F, 0, 0), L, L)

    for u in 2:L
        Au = trace(A[u-1])
        for t in 1:u-2
            M[t, u] = M[t, u-1] * Au
        end
        # initial condition
        M[u-1, u] = Matrix{F}(I, size(A[u],1), size(A[u],1))
    end

    return M
end

"""
    lognormalization(A::AbstractTensorTrain)

Compute the natural logarithm of the normalization ``\\log Z=\\log \\sum_{x^1,\\ldots,x^L} A^1(x^1)\\cdots A^L(x^L)``.
Throws an error if the normalization is negative.
"""
function lognormalization(A::AbstractTensorTrain)
    return log(normalization(A))
end

"""
    normalization(A::AbstractTensorTrain)

Compute the normalization ``Z=\\sum_{x^1,\\ldots,x^L} A^1(x^1)\\cdots A^L(x^L)`` and return it as a `Logarithmic`, which stores the sign and the logarithm of the absolute value (see the docs of LogarithmicNumbers.jl https://github.com/cjdoris/LogarithmicNumbers.jl?tab=readme-ov-file#documentation)
"""
function normalization(A::AbstractTensorTrain)
    l, z = accumulate_L(A)
    return z / A.z
end

"""
    normalize_eachmatrix!(A::AbstractTensorTrain)

Divide each matrix by its maximum (absolute) element and return the sum of the logs of the individual normalizations.
This is used to keep the entries from exploding during computations
"""
function normalize_eachmatrix!(A::AbstractTensorTrain{F}) where F
    c = Logarithmic(one(F))
    for m in A
        mm = maximum(abs, m)
        if !isnan(mm) && !isinf(mm) && !iszero(mm)
            m ./= mm
            c *= mm
        end
    end
    A.z /= c
    return nothing
end

"""
    LinearAlgebra.normalize!(A::AbstractTensorTrain)

Compute the normalization of ``Z=\\sum_{x^1,\\ldots,x^L} A^1(x^1)\\cdots A^L(x^L)`` (see [`normalization`](@ref)) and rescale the tensors in `A` such that, after this call, ``|Z|=1``.
Return the natural logarithm of the absolute normalization ``\\log|Z|``
"""
function LinearAlgebra.normalize!(A::AbstractTensorTrain)
    absZ = abs(accumulate_L(A)[2])
    L = length(A)
    x = exp(1/L * log(absZ))
    if x != 0
        for a in A
            a ./= x
        end
    end
    logz = log(absZ / A.z)
    A.z = 1
    return logz
end


"""
    marginals(A::AbstractTensorTrain; l, r)

Compute the marginal distributions ``p(x^l)`` at each site

### Optional arguments
- `l = accumulate_L(A)[1]`, `r = accumulate_R(A)[1]` pre-computed partial normalizations
"""
function marginals(A::AbstractTensorTrain{F,N};
    l = accumulate_L(A)[1], r = accumulate_R(A)[1]) where {F<:Real,N}

    map(eachindex(A)) do t 
        Aᵗ = _reshape1(A[t])
        R = t + 1 ≤ length(A) ? r[t+1] : Matrix(I, size(A[end],2), size(A[end],2))
        L = t - 1 ≥ 1 ? l[t-1] : Matrix(I, size(A[begin],1), size(A[begin],1))
        @tullio lA[a¹,aᵗ⁺¹,x] := L[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x]
        @tullio pᵗ[x] := lA[a¹,aᵗ⁺¹,x] * R[aᵗ⁺¹,a¹]
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end
end

"""
    twovar_marginals(A::AbstractTensorTrain; l, r, M, Δlmax)

Compute the marginal distributions for each pair of sites ``p(x^l, x^m)``

### Optional arguments
- `l = accumulate_L(A)[1]`, `r = accumulate_R(A)[1]`, `M = accumulate_M(A)` pre-computed partial normalizations
- `maxdist = length(A)`: compute marginals only at distance `maxdist`: ``|l-m|\\le maxdist``
"""
function twovar_marginals(A::AbstractTensorTrain{F,N};
    l = accumulate_L(A)[1], r = accumulate_R(A)[1], M = accumulate_M(A),
    maxdist = length(A)-1) where {F<:Real,N}
    qs = tuple(reduce(vcat, [x,x] for x in size(A[begin])[3:end])...)
    b = Array{F,2*(N-2)}[zeros(zeros(Int, 2*(N-2))...) 
        for _ in eachindex(A), _ in eachindex(A)]
    d = first(bond_dims(A))
    for t in 1:length(A)-1
        lᵗ⁻¹ = t == 1 ? Matrix(I, d, d) : l[t-1]
        Aᵗ = _reshape1(A[t])
        for u in t+1:min(length(A),t+maxdist)
            rᵘ⁺¹ = u == length(A) ? Matrix(I, d, d) : r[u+1]
            Aᵘ = _reshape1(A[u])
            Mᵗᵘ = M[t, u]
            rl = rᵘ⁺¹ * lᵗ⁻¹
            @tullio rlAt[aᵘ⁺¹, aᵗ⁺¹, xᵗ] := rl[aᵘ⁺¹,aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ]
            @tullio rlAtMtu[aᵘ⁺¹,xᵗ,aᵘ] := rlAt[aᵘ⁺¹, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ]
            @tullio bᵗᵘ[xᵗ, xᵘ] := rlAtMtu[aᵘ⁺¹,xᵗ,aᵘ] * Aᵘ[aᵘ, aᵘ⁺¹, xᵘ]
            bᵗᵘ ./= sum(bᵗᵘ)
            b[t,u] = reshape(bᵗᵘ, qs)
        end
    end
    b
end

"""
    compress!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Compress `A` by means of SVD decompositions + truncations
"""
function compress!(A::AbstractTensorTrain; svd_trunc=TruncThresh(1e-6),
        is_orthogonal::Symbol=:none)
    if is_orthogonal == :none
        orthogonalize_right!(A; svd_trunc=TruncThresh(0.0))
        orthogonalize_left!(A; svd_trunc)
    elseif is_orthogonal == :left
        orthogonalize_right!(A; svd_trunc)
    elseif is_orthogonal == :right
        orthogonalize_left!(A; svd_trunc)
    else
        throw(ArgumentError("Keyword `is_orthogonal` only supports: :none, :left, :right, got :$is_orthogonal"))
    end
    return A
end

"""
    Base.:(+)(A::AbstracTensorTrain, B::AbstracTensorTrain)

Compute the sum of two Tensor Trains. Matrix sizes are doubled
"""
Base.:(+)(A::AbstractTensorTrain, B::AbstractTensorTrain) = _compose(+, A, B)

"""
    Base.:(-)(A::AbstracTensorTrain, B::AbstracTensorTrain)

Compute the difference of two Tensor Trains. Matrix sizes are doubled
"""
Base.:(-)(A::AbstractTensorTrain, B::AbstractTensorTrain) = _compose(-, A, B)

"""
    sample([rng], A::AbstractTensorTrain; r)

Draw an exact sample from `A` interpreted as a probability distribution.
`A` doesn't need to be normalized, however error will be raised if it is found to take negative values.
    
Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `rz = accumulate_R(A)`.

The output is `x, p`, the sampled sequence and its probability
"""
function StatsBase.sample(rng::AbstractRNG, A::AbstractTensorTrain{F,N};
        rz = accumulate_R(A)) where {F<:Real,N}
    x = [zeros(Int, N-2) for Aᵗ in A]
    sample!(rng, x, A; rz)
end
function StatsBase.sample(A::AbstractTensorTrain{F,N}; rz = accumulate_R(A)) where {F<:Real,N}
    sample(default_rng(), A; rz)
end

"""
    sample!([rng], x, A::AbstractTensorTrain; r)

Draw an exact sample from `A` interpreted as a probability distribution and store the result in `x`.
`A` doesn't need to be normalized, however error will be raised if it is found to take negative values.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `rz = accumulate_R(A)`.

The output is `x, p`, the sampled sequence and its probability
"""
function StatsBase.sample!(rng::AbstractRNG, x, A::AbstractTensorTrain{F,N};
        rz = accumulate_R(A)) where {F<:Real,N}
    r, z = rz
    L = length(A)
    @assert length(x) == L
    @assert all(length(xᵗ) == N-2 for xᵗ in x)
    d = first(bond_dims(A))

    Q = Matrix(I, d, d)     # stores product of the first `t` matrices, evaluated at the sampled `x¹,...,xᵗ`
    for t in eachindex(A)
        rᵗ⁺¹ = t == L ? Matrix(I, d, d) : r[t+1]
        # collapse multivariate xᵗ into 1D vector, sample from it
        Aᵗ = _reshape1(A[t])
        @tullio QA[k,n,x] := Q[k,m] * Aᵗ[m,n,x]
        @tullio p[x] := QA[k,n,x] * rᵗ⁺¹[n,k]
        if any(<(0), p)
            error("Cannot sample from a tensor train with negative values")
        end
        p ./= sum(p)
        xᵗ = sample_noalloc(rng, p)
        x[t] .= CartesianIndices(size(A[t])[3:end])[xᵗ] |> Tuple
        # update prob
        Q = Q * Aᵗ[:,:,xᵗ]
    end
    p = float(tr(Q) / z)
    return x, p
end
function StatsBase.sample!(x, A::AbstractTensorTrain{F,N}; rz = accumulate_R(A)) where {F<:Real,N}
    sample!(default_rng(), x, A; rz)
end

@doc raw"""
    LinearAlgebra.dot(A::AbstractTensorTrain, B::AbstractTensorTrain)

Compute the inner product between tensor trains `A` and `B`

```math
A\cdot B = \sum_{x^1,x^2,\ldots,x^L}A^1(x^1)A^2(x^2)\cdots A^L(x^L)B^1(x^1)B^2(x^2)\cdots B^L(x^L)
```
"""
function LinearAlgebra.dot(A::AbstractTensorTrain, B::AbstractTensorTrain)
    Aᴸ = _reshape1(A[end])
    Bᴸ = _reshape1(B[end])
    @tullio C[aᴸ,a¹,b¹,bᴸ] := Aᴸ[aᴸ,a¹,xᴸ] * conj(Bᴸ[bᴸ,b¹,xᴸ])

    for (Al, Bl) in Iterators.drop(Iterators.reverse(zip(A,B)), 1)
        Aˡ = _reshape1(Al)
        Bˡ = _reshape1(Bl)
        C = @tullio _[aˡ,a¹,b¹,bˡ] := Aˡ[aˡ,aˡ⁺¹,xˡ] * C[aˡ⁺¹,a¹,b¹,bˡ⁺¹] * conj(Bˡ[bˡ,bˡ⁺¹,xˡ])
    end

    @tullio d = C[a¹,a¹,b¹,b¹]
    return d / float(A.z * B.z)
end

@doc raw"""
    LinearAlgebra.norm(A::AbstractTensorTrain)

Compute the 2-norm (Frobenius norm) of tensor train `A`

```math
\lVert A\rVert_2 = \sqrt{\sum_{x^1,x^2,\ldots,x^L}\left[A^1(x^1)A^2(x^2)\cdots A^L(x^L)\right]^2} = \sqrt{A\cdot A}
```
"""
LinearAlgebra.norm(A::AbstractTensorTrain) = sqrt(dot(A, A))

@doc raw"""
    norm2m(A::AbstractTensorTrain, B::AbstractTensorTrain)

Given two tensor trains `A,B`, compute `norm(A - B)^2` as

```math
\lVert A-B\rVert_2^2 = \lVert A \rVert_2^2 + \lVert B \rVert_2^2 - 2A\cdot B
```
"""
function norm2m(A::AbstractTensorTrain, B::AbstractTensorTrain) 
    return norm(A)^2 + norm(B)^2 - 2*real(dot(A, B))
end