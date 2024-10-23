"""
    AbstractTransferOperator{TA,TM}

A type to represent a transfer operator.
"""
abstract type AbstractTransferOperator{TA<:Number,TM<:Number} end

@doc raw"""
    TransferOperator{TA<:Number,TM<:Number} <: AbstractTransferOperator{TA,TM}

A type to represent a transfer operator $G$ obtained from variable-dependent matrices $A(x)$, $M(x)$ as
```math
G_{i,j,k,l} = \sum_x A_{i,k}(x) M^*_{j,l}(x)
    ```
"""
struct TransferOperator{TA<:Number,TM<:Number} <: AbstractTransferOperator{TA,TM}
    A :: Array{TA,3}
    M :: Array{TM,3}
end

@doc raw"""
    HomogeneousTransferOperator{T<:Number} <: AbstractTransferOperator{T,T}

A type to represent a transfer operator $G$ obtained from variable-dependent matrix $A(x)$ as
```math
G_{i,j,k,l} = \sum_x A_{i,k}(x) A^*_{j,l}(x)
    ```
"""
struct HomogeneousTransferOperator{T<:Number} <: AbstractTransferOperator{T,T}
    A :: Array{T,3}
end

function sizes(G::AbstractTransferOperator)
    A, M = get_tensors(G)
    size(A, 1), size(M, 1), size(A, 2), size(M, 2)
end

get_tensors(G::TransferOperator) = (G.A, G.M)
get_tensors(G::HomogeneousTransferOperator) = (G.A, G.A)

function Base.convert(::Type{TransferOperator}, G::HomogeneousTransferOperator)
    TransferOperator(get_tensors(G)...)
end
TransferOperator(G::HomogeneousTransferOperator) = convert(TransferOperator, G)

function transfer_operator(q::AbstractUniformTensorTrain, p::AbstractUniformTensorTrain)
    return TransferOperator(_reshape1(q.tensor) / float(q.z), _reshape1(p.tensor) / float(p.z))
end
function transfer_operator(q::AbstractUniformTensorTrain)
    return HomogeneousTransferOperator(_reshape1(q.tensor) / float(q.z))
end

function Base.collect(G::AbstractTransferOperator)
    A, M = get_tensors(G)
    return @tullio B[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
end

function Base.:(*)(G::AbstractTransferOperator, B::AbstractMatrix)
    A, M = get_tensors(G)
    return @tullio C[i,j] := A[i,k,x] * conj(M[j,l,x]) * B[k,l]
end

function Base.:(*)(B::AbstractMatrix, G::AbstractTransferOperator)
    A, M = get_tensors(G)
    return @tullio C[k,l] := B[i,j] * A[i,k,x] * conj(M[j,l,x])
end

function leading_eig(G::AbstractTransferOperator)
    GG = collect(G)
    @cast B[(i,j),(k,l)] := GG[i,j,k,l]
    valsR, vecsR = eigsolve(B)
    valsL, vecsL = eigsolve(B')
    valsR[1] ≈ valsL[1] || @warn "Leading eigenvalue for A and Aᵀ not equal, got $(valsR[1]) and $(valsL[1])"
    λ = complex(valsL[1])
    L = vecsL[1]
    R = vecsR[1]
    d = sizes(G)
    r = reshape(R, d[1], d[2])
    l = reshape(L, d[1], d[2])
    l ./= dot(l, r)
    return λ, l, r
end

"""
    dot(q::InfiniteUniformTensorTrain, p::InfiniteUniformTensorTrain)

Return the "dot product" between the infinite tensor trains `q` and `p`.

Since two infinite tensor trains are either equal or orthogonal (orthogonality catastrophe), what is actually returned here is the leading eigenvalue of the transfer operator obtained from the matrices of `q` and `p`
"""
function LinearAlgebra.dot(q::InfiniteUniformTensorTrain, p::InfiniteUniformTensorTrain)
    G = transfer_operator(q, p)
    Eq = transfer_operator(q)
    Ep = transfer_operator(p)
    λG, = leading_eig(G)
    λq, = leading_eig(Eq)
    λp, = leading_eig(Ep)
    return λG / sqrt(λp * λq)
end