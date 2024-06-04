"""
    AbstractUniformTensorTrain{F,N} <: AbstractPeriodicTensorTrain{F,N}

An abstract type for representing tensor trains with periodic boundary conditions and all matrices equal
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
abstract type AbstractUniformTensorTrain{F,N} <: AbstractPeriodicTensorTrain{F,N} end

"""
    UniformTensorTrain{F<:Number, N} <: AbstractUniformTensorTrain{F,N}

A type for representing a tensor train with periodic boundary conditions, all matrices equal and a fixed length `L`.
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)

## FIELDS
- `tensor` only one is stored
- `L` the length of the tensor train
- `z` a re-scaling constant
"""
mutable struct UniformTensorTrain{F<:Number, N} <: AbstractUniformTensorTrain{F,N}
    tensor::Array{F,N}
    L :: Int
    z :: Logarithmic{F}

    function UniformTensorTrain{F,N}(tensor::Array{F,N}, L::Integer; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensor,1) == size(tensor,2) || throw(ArgumentError("Matrix must be square"))
        L > 0 || throw(ArgumentError("Length `L` must be positive, got $L"))
        return new{F,N}(tensor, Int(L), z)
    end
end
function UniformTensorTrain(tensor::Array{F,N}, L::Integer; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N} 
    return UniformTensorTrain{F,N}(tensor, L; z)
end

"""
    periodic_tensor_train(A::UniformTensorTrain)

Produce a `PeriodicTensorTrain` corresponding to `A`, with the matrix concretely repeated `length(A)` times
"""
periodic_tensor_train(A::UniformTensorTrain) = PeriodicTensorTrain(fill(A.tensor, A.L); z = A.z)

Base.length(A::UniformTensorTrain) = A.L

function Base.getindex(A::UniformTensorTrain, i::Integer)
    L = length(A)
    i in 1:L || throw(BoundsError("attempt to access $L-element UniformTensorTrain at index $i"))
    return A.tensor
end

Base.iterate(A::UniformTensorTrain, i=1) = (@inline; (i % UInt) - 1 < length(A) ? (@inbounds A[i], i + 1) : nothing)
Base.firstindex(A::UniformTensorTrain) = 1
Base.lastindex(A::UniformTensorTrain) = length(A)
Base.eachindex(A::UniformTensorTrain) = 1:length(A)

Base.:(==)(A::T, B::T) where {T<:UniformTensorTrain} = isequal(A.L, B.L) && isequal(A.tensor, B.tensor)
Base.isapprox(A::T, B::T; kw...) where {T<:UniformTensorTrain} = isequal(A.L, B.L) && isapprox(A.tensor, B.tensor; kw...)

function Base.setindex!(::UniformTensorTrain, x, i::Integer)
    throw(ArgumentError("Cannot setindex! to $i for a UniformTensorTrain"))
end

# computes B = ∑ₓA(x)
function one_normalization(A::AbstractUniformTensorTrain{F,N}) where {F,N}
    dims = tuple(3:N...)
    B = Matrix(dropdims(sum(A.tensor; dims); dims))
    return B
end

function TensorTrains.normalization(A::UniformTensorTrain; B = one_normalization(A))
    L = length(A)
    return tr(B^L) / A.z
end

function LinearAlgebra.normalize!(A::UniformTensorTrain)
    L = length(A)
    Z = abs(tr(one_normalization(A)^L))
    A.tensor ./= Z^(1/L)
    logz = log(Z / A.z)
    A.z = 1
    return logz
end

function TensorTrains.marginals(A::UniformTensorTrain; B = one_normalization(A))
    L = length(A)
    C = B^(L-1)
    m = map(Iterators.product(axes(A.tensor)[3:end]...)) do x
        tr(C * @view A.tensor[:,:,x...])
    end
    return [m / sum(m)]
end

function TensorTrains.orthogonalize_left!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function TensorTrains.orthogonalize_right!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function TensorTrains.compress!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function TensorTrains._compose(f, ::UniformTensorTrain, ::UniformTensorTrain)
    error("Not implemented")
end

function Base.:(+)(A::UniformTensorTrain{F,NA}, B::UniformTensorTrain{F,NB}) where {F,NA,NB}
    NA == NB || throw(ArgumentError("Tensor Trains must have the same number of variables, got $NA and $NB"))
    L = length(A)
    @assert length(B) == L
    sa = size(A.tensor); sb = size(B.tensor)
    C = [ [A.tensor[:,:,x...]/float(A.z^(1/L)) zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) B.tensor[:,:,x...]/float(B.z^(1/L))]
                for x in Iterators.product(axes(A.tensor)[3:end]...)]
    tensor = reshape( reduce(hcat, C), (sa .+ sb)[1:2]..., size(A.tensor)[3:end]... )
    return UniformTensorTrain(tensor, L)
end

@doc raw"""
    symmetrized_uniform_tensor_train(A::AbstractTensorTrain)

Convert a tensor train ``f(x^1,x^2,\ldots,x^L)`` into a `UniformTensorTrain` ``g`` such that

```math
g(\mathcal P[x^1,x^2,\ldots,x^L]) = g(x^1,x^2,\ldots,x^L) = \sum_{l=1}^L f(x^l, x^{l+1},\ldots,x^L,x^1,\ldots,x^{l-1})
```

for any cyclic permutation ``\mathcal P``
"""
function symmetrized_uniform_tensor_train(A::AbstractTensorTrain)
    sz = size(A[1])[3:end]
    rowdims = [size(a, 1) for a in A]
    coldims = [size(a, 2) for a in A]
    nstates = [1:s for s in sz]
    tensor = zeros(sum(rowdims), sum(coldims), sz...)
    for x in Iterators.product(nstates...)
        for i in eachindex(A)
            r = sum(rowdims[1:i-1])
            c = sum(coldims[1:i-1])
            tensor[r+1:r+rowdims[i],c+1:c+coldims[i],x...] = A[i][:,:,x...]
            tensor[:,c+1:c+coldims[i],x...] .= circshift(tensor[:,c+1:c+coldims[i],x...], (-rowdims[1],0))
        end
    end
    return UniformTensorTrain(tensor, length(A); z = A.z)
end

"""
    InfiniteUniformTensorTrain{F<:Number, N} <: AbstractUniformTensorTrain{F,N}

A type for representing an infinite tensor train with periodic boundary conditions and all matrices equal.
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)

## FIELDS
- `tensor` only one is stored
- `z` a re-scaling constant
"""
mutable struct InfiniteUniformTensorTrain{F<:Number, N} <: AbstractUniformTensorTrain{F,N}
    tensor::Array{F,N}
    z :: Logarithmic{F}

    function InfiniteUniformTensorTrain{F,N}(tensor::Array{F,N}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensor,1) == size(tensor,2) || throw(ArgumentError("Matrix must be square"))
        return new{F,N}(tensor, z)
    end
end
function InfiniteUniformTensorTrain(tensor::Array{F,N}; z::Logarithmic{F}=Logarithmic(one(F))) where {F<:Number, N} 
    return InfiniteUniformTensorTrain{F,N}(tensor; z)
end

function flat_infinite_uniform_tt(d::Integer, q...)
    x = 1 / (d * prod(q))
    tensor = fill(x, d, d, q...)
    return InfiniteUniformTensorTrain(tensor)
end
function rand_infinite_uniform_tt(d::Integer, q...)
    tensor = rand(d, d, q...)
    return InfiniteUniformTensorTrain(tensor)
end

Base.:(==)(A::T, B::T) where {T<:InfiniteUniformTensorTrain} = isequal(A.tensor, B.tensor)
Base.isapprox(A::T, B::T; kw...) where {T<:InfiniteUniformTensorTrain} = isapprox(A.tensor, B.tensor; kw...)

function _eigen(A::InfiniteUniformTensorTrain; B = one_normalization(A))
    d, R, _ = eigsolve(B)
    _, L, _ = eigsolve(B')
    r = R[1]
    l = L[1]
    # normalize such that <l|r>=1
    l ./= dot(l, r)
    λ = d[1]
    λ, l, r
end

function TensorTrains.normalization(A::InfiniteUniformTensorTrain; B = one_normalization(A))
    λ, = _eigen(A; B)
    return λ / A.z
end

function LinearAlgebra.normalize!(A::InfiniteUniformTensorTrain)
    B = one_normalization(A)
    λ, = _eigen(A; B)
    Z = abs(λ)
    A.tensor ./= Z
    logz = log(Z / A.z)
    A.z = 1
    return logz
end

function Base.:(+)(A::InfiniteUniformTensorTrain{F,NA}, B::InfiniteUniformTensorTrain{F,NB}) where {F,NA,NB}
    NA == NB || throw(ArgumentError("Tensor Trains must have the same number of variables, got $NA and $NB"))
    sa = size(A.tensor); sb = size(B.tensor)
    C = [ [A.tensor[:,:,x...] zeros(sa[1],sb[2])/float(A.z); zeros(sb[1],sa[2]) B.tensor[:,:,x...]/float(B.z)] 
                for x in Iterators.product(axes(A.tensor)[3:end]...)]
    tensor = reshape( reduce(hcat, C), (sa .+ sb)[1:2]..., size(A.tensor)[3:end]...)
    return InfiniteUniformTensorTrain(tensor)
end

function TensorTrains.marginals(A::InfiniteUniformTensorTrain; B = one_normalization(A))
    _, l, r = _eigen(A; B)
    iter = Iterators.product(axes(A.tensor)[3:end]...)
    m = map(iter) do x
        l' * (@view A.tensor[:,:,x...]) * r
    end
    m ./= sum(m)
    return [m]
end

function TensorTrains.twovar_marginals(A::InfiniteUniformTensorTrain{F}; B = one_normalization(A)) where F
    _, l, r = _eigen(A; B)
    iter = Iterators.product(axes(A.tensor)[3:end]...)
    m = map(Iterators.product(iter, iter)) do (x1, x2)
        l' * (@views A.tensor[:,:,x1...] * A.tensor[:,:,x2...]) * r
    end
    m ./= sum(m)
    return [m]
end

function TensorTrains.normalize_eachmatrix!(A::InfiniteUniformTensorTrain{F}) where {F}
    c = Logarithmic(one(F))
    mm = maximum(abs, A.tensor)
    if !isnan(mm) && !isinf(mm) && !iszero(mm)
        A.tensor ./= mm
        c *= mm
    end
    A.z /= c
    return nothing
end