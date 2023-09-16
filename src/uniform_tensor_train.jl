struct UniformTensorTrain{F<:Number, N} <: AbstractPeriodicTensorTrain{F,N}
    tensor::Array{F,N}
    L :: Int

    function UniformTensorTrain{F,N}(tensor::Array{F,N}, L::Integer) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensor,1) == size(tensor,2) ||
            throw(ArgumentError("Matrix must be square"))
        L > 0 || throw(ArgumentError("Length `L` must be positive, got $L"))
        return new{F,N}(tensor, Int(L))
    end
end
function UniformTensorTrain(tensor::Array{F,N}, L::Integer) where {F<:Number, N} 
    return UniformTensorTrain{F,N}(tensor, L)
end

periodic_tensor_train(A::UniformTensorTrain) = PeriodicTensorTrain(fill(A.tensor, A.L))

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
function one_normalization(A::UniformTensorTrain{F,N}) where {F,N}
    dims = tuple(3:N...)
    B = Matrix(dropdims(sum(A.tensor; dims); dims))
    return B
end

function normalization(A::UniformTensorTrain; B = one_normalization(A))
    L = length(A)
    return tr(B^L)
end

function marginals(A::UniformTensorTrain; B = one_normalization(A))
    L = length(A)
    C = B^(L-1)
    m = map(Iterators.product(axes(A.tensor)[3:end]...)) do x
        tr(C * @view A.tensor[:,:,x...])
    end
    return [m / sum(m)]
end

function orthogonalize_left!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function orthogonalize_right!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function compress!(::UniformTensorTrain; svd_trunc = TruncThresh(0.0))
    error("Not implemented")
end

function _compose(f, ::UniformTensorTrain, ::UniformTensorTrain)
    error("Not implemented")
end

function Base.:(+)(A::UniformTensorTrain{F,NA}, B::UniformTensorTrain{F,NB}) where {F,NA,NB}
    NA == NB || throw(ArgumentError("Tensor Trains must have the same number of variables, got $NA and $NB"))
    L = length(A)
    @assert length(B) == L
    sa = size(A.tensor); sb = size(B.tensor)
    C = [ [A.tensor[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) B.tensor[:,:,x...]] 
                for x in Iterators.product(axes(A.tensor)[3:end]...)]
    tensor = reshape( reduce(hcat, C), (sa .+ sb)[1:2]..., size(A.tensor)[3:end]...)
    return UniformTensorTrain(tensor, L)
end

# function Base.:(-)(::UniformTensorTrain, ::UniformTensorTrain)
#     error("Not implemented")
# end

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
    return UniformTensorTrain(tensor, length(A))
end