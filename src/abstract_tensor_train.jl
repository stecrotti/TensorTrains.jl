"""
    AbstractTensorTrain

An abstract type representing a Tensor Train.
It currently supports 2 subtypes [`TensorTrain`](@ref) and [`PeriodicTensorTrain`](@ref).
"""
abstract type AbstractTensorTrain{F<:Number, N} end

Base.eltype(::AbstractTensorTrain{F,N}) where {N,F} = F

"""
    normalize_eachmatrix!(A::AbstractTensorTrain)

Divide each matrix by its maximum (absolute) element and return the sum of the logs of the individual normalizations.
This is used to keep the entries from exploding during computations
"""
function normalize_eachmatrix!(A::AbstractTensorTrain)
    c = 0.0
    for m in A
        mm = maximum(abs, m)
        if !any(isnan, mm) && !any(isinf, mm)
            m ./= mm
            c += log(mm)
        end
    end
    c
end

Base.:(==)(A::T, B::T) where {T<:AbstractTensorTrain} = isequal(A.tensors, B.tensors)
Base.isapprox(A::T, B::T; kw...) where {T<:AbstractTensorTrain} = isapprox(A.tensors, B.tensors; kw...)


function accumulate_M(A::AbstractTensorTrain)
    L = length(A)
    M = [zeros(0, 0) for _ in 1:L, _ in 1:L]
    
    # initial condition
    for t in 1:L-1
        range_aᵗ⁺¹ = axes(A[t+1], 1)
        Mᵗᵗ⁺¹ = [float((a == c)) for a in range_aᵗ⁺¹, c in range_aᵗ⁺¹]
        M[t, t+1] = Mᵗᵗ⁺¹
    end

    for t in 1:L-1
        Mᵗᵘ⁻¹ = M[t, t+1]
        for u in t+2:L
            Aᵘ⁻¹ = _reshape1(A[u-1])
            @tullio Mᵗᵘ[aᵗ⁺¹, aᵘ] := Mᵗᵘ⁻¹[aᵗ⁺¹, aᵘ⁻¹] * Aᵘ⁻¹[aᵘ⁻¹, aᵘ, x]
            M[t, u] = Mᵗᵘ
            Mᵗᵘ⁻¹, Mᵗᵘ = Mᵗᵘ, Mᵗᵘ⁻¹
        end
    end

    return M
end


"""
    compress!(A::AbstractTensorTrain; svd_trunc::SVDTrunc)

Compress `A` by means of SVD decompositions + truncations
"""
function compress!(A::AbstractTensorTrain; svd_trunc=TruncThresh(1e-6))
    orthogonalize_right!(A, svd_trunc=TruncThresh(0.0))
    orthogonalize_left!(A; svd_trunc)
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

Draw an exact sample from `A`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.GLOBAL_RNG`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function StatsBase.sample(rng::AbstractRNG, A::AbstractTensorTrain{F,N};
        r = accumulate_R(A)) where {F<:Real,N}
    x = [zeros(Int, N-2) for Aᵗ in A]
    sample!(rng, x, A; r)
end
function StatsBase.sample(A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample(GLOBAL_RNG, A; r)
end

"""
    sample!([rng], x, A::AbstractTensorTrain; r)

Draw an exact sample from `A` and store the result in `x`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.GLOBAL_RNG`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function StatsBase.sample!(x, A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample!(GLOBAL_RNG, x, A; r)
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
    @tullio C[aᴸ,a¹,b¹,bᴸ] := Aᴸ[aᴸ,a¹,xᴸ] * Bᴸ[bᴸ,b¹,xᴸ]

    for (Al, Bl) in Iterators.drop(Iterators.reverse(zip(A,B)), 1)
        Aˡ = _reshape1(Al)
        Bˡ = _reshape1(Bl)
        @tullio Cnew[aˡ,a¹,b¹,bˡ] := Aˡ[aˡ,aˡ⁺¹,xˡ] * C[aˡ⁺¹,a¹,b¹,bˡ⁺¹] * Bˡ[bˡ,bˡ⁺¹,xˡ]
        C = Cnew
    end

    @tullio d = C[a¹,a¹,b¹,b¹]
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
    return norm(A)^2 + norm(B)^2 - 2*dot(A, B)
end

"""
    LinearAlgebra.normalize!(A::AbstractTensorTrain)

Normalize `A` to a probability distribution
"""
function LinearAlgebra.normalize!(A::AbstractTensorTrain)
    c = normalize_eachmatrix!(A)
    Z = normalization(A)
    L = length(A)
    for a in A
        a ./= Z^(1/L)
    end
    c + log(Z)
end