"""
    AbstractTensorTrain

An abstract type representing a Tensor Train.
It currently supports 2 subtypes [`TensorTrain`](@ref) and [`PeriodicTensorTrain`](@ref).
"""
abstract type AbstractTensorTrain{F<:Number, N} end

eltype(::AbstractTensorTrain{F,N}) where {N,F} = F

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

==(A::T, B::T) where {T<:AbstractTensorTrain} = isequal(A.tensors, B.tensors)
isapprox(A::T, B::T; kw...) where {T<:AbstractTensorTrain} = isapprox(A.tensors, B.tensors; kw...)


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
    +(A::AbstracTensorTrain, B::AbstracTensorTrain)

Compute the sum of two Tensor Trains. Matrix sizes are doubled
"""
+(A::AbstractTensorTrain, B::AbstractTensorTrain) = _compose(+, A, B)

"""
    -(A::AbstracTensorTrain, B::AbstracTensorTrain)

Compute the difference of two Tensor Trains. Matrix sizes are doubled
"""
-(A::AbstractTensorTrain, B::AbstractTensorTrain) = _compose(-, A, B)

## Fallback sampling methods
function sample!(x, A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample!(GLOBAL_RNG, x, A; r)
end

@doc raw"""
    LinearAlgebra.norm(A::AbstractTensorTrain)

Compute the 2-norm (Frobenius norm) of tensor train `A`

```math
\sqrt{\sum_x\left|A(x)\right|_2^2} = \sqrt{\sum_x \text{Tr}\left[A(x)A(x)^\dagger\right]}
```
"""
function norm(A::AbstractTensorTrain)
    Aᵀ = _reshape1(A[end])
    R = sum(Aᵀ[:,:,xᵀ] * Aᵀ[:,:,xᵀ]' for xᵀ in axes(Aᵀ, 3))
    Aᵀ = reshape(A[end], size(A[end], 1), :)
    for At in Iterators.drop(Iterators.reverse(A), 1)
        Aᵗ = _reshape1(At)
        R = Hermitian(sum(Aᵗ[:,:,xᵗ] * R * Aᵗ[:,:,xᵗ]' for xᵗ in axes(Aᵗ, 3)))
    end
    sqrt(tr(R))
end

"""
    sample([rng], A::AbstractTensorTrain; r)

Draw an exact sample from `A`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.GLOBAL_RNG`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function sample(rng::AbstractRNG, A::AbstractTensorTrain{F,N};
        r = accumulate_R(A)) where {F<:Real,N}
    x = [zeros(Int, N-2) for Aᵗ in A]
    sample!(rng, x, A; r)
end
function sample(A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample(GLOBAL_RNG, A; r)
end