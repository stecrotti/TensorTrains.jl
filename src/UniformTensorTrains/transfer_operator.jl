abstract type AbstractTransferOperator{F1<:Number,F2<:Number} end
abstract type AbstractFiniteTransferOperator{F1<:Number,F2<:Number} <: AbstractTransferOperator{F1,F2} end

struct TransferOperator{F1<:Number,F2<:Number} <: AbstractFiniteTransferOperator{F1,F2}
    A :: Array{F1,3}
    M :: Array{F2,3}
end

struct HomogeneousTransferOperator{F<:Number} <: AbstractFiniteTransferOperator{F,F}
    A :: Array{F,3}
end

function sizes(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    size(A, 1), size(M, 1), size(A, 2), size(M, 2)
end

get_tensors(G::TransferOperator) = (G.A, G.M)
get_tensors(G::HomogeneousTransferOperator) = (G.A, G.A)

function Base.convert(::Type{TransferOperator}, G::HomogeneousTransferOperator)
    TransferOperator(get_tensors(G)...)
end
TransferOperator(G::HomogeneousTransferOperator) = convert(TransferOperator, G)

# the first argument `p` is the one with `A` matrices
function transfer_operator(q::AbstractUniformTensorTrain, p::AbstractUniformTensorTrain)
    return TransferOperator(_reshape1(q.tensor), _reshape1(p.tensor))
end
function transfer_operator(q::AbstractUniformTensorTrain)
    return HomogeneousTransferOperator(_reshape1(q.tensor))
end

function Base.collect(G::AbstractFiniteTransferOperator)
    A, M = get_tensors(G)
    return @tullio B[i,j,k,l] := A[i,k,x] * conj(M[j,l,x])
end

function Base.:(*)(G::AbstractFiniteTransferOperator, B::AbstractMatrix)
    A, M = get_tensors(G)
    return @tullio C[i,j] := A[i,k,x] * conj(M[j,l,x]) * B[k,l]
end

function Base.:(*)(B::AbstractMatrix, G::AbstractFiniteTransferOperator)
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
    return (; l, r, λ)
end


struct InfiniteTransferOperator{F<:Number,M<:AbstractMatrix{F}} <: AbstractTransferOperator{F,F}
    l :: M
    r :: M
    λ :: F
end

function infinite_transfer_operator(G::AbstractTransferOperator; lambda1::Bool=false)
    l, r, λ_ = leading_eig(G)
    λ = lambda1 ? one(λ_) : λ_
    λ = convert(eltype(r), λ)
    InfiniteTransferOperator(l, r, λ)
end

function Base.collect(G::InfiniteTransferOperator)
    (; l, r, λ) = G
    return @tullio B[i,j,k,m] := r[i,j] * l[k,m]
end

function sizes(G::InfiniteTransferOperator)
    (; l, r) = G
    return tuple(size(r)..., size(l)...)
end

function infinite_transfer_operator(q::AbstractUniformTensorTrain, p::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(q, p))
end

function infinite_transfer_operator(q::AbstractUniformTensorTrain)
    return infinite_transfer_operator(transfer_operator(q))
end

function LinearAlgebra.dot(q::InfiniteUniformTensorTrain, p::InfiniteUniformTensorTrain;
        G = infinite_transfer_operator(q, p),
        Ep = infinite_transfer_operator(p),
        Eq = infinite_transfer_operator(q))
    return G.λ / sqrt(abs(Ep.λ * Eq.λ))
end