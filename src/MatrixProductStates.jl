module MatrixProductStates

using TensorTrains
import TensorTrains: _reshape1, accumulate_L, accumulate_R, sample_noalloc
using Lazy: @forward
using Tullio: @tullio
using Random: AbstractRNG, default_rng
using StatsBase
using LinearAlgebra: I, tr

export MatrixProductState

struct MatrixProductState{T<:AbstractTensorTrain}
    ψ :: T
end

@forward MatrixProductState.ψ bond_dims, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, check_bond_dims, Base.length, Base.eachindex

Base.:(==)(A::T, B::T) where {T<:MatrixProductState} = isequal(A.ψ, B.ψ)
Base.isapprox(A::T, B::T; kw...) where {T<:MatrixProductState} = isapprox(A.ψ, B.ψ; kw...)


TensorTrains.bond_dims(p::MatrixProductState) = bond_dims(p.ψ)

TensorTrains.evaluate(p::MatrixProductState, X...) = abs2(evaluate(p.ψ, X...))

id4(d::Integer) = [a==a¹ && b==b¹ for a in 1:d, a¹ in 1:d, b in 1:d, b¹ in 1:d]

function TensorTrains.accumulate_L(p::MatrixProductState)
    (; ψ) = p
    d = size(ψ[begin], 1)
    L = id4(d)
    return map(_reshape1(Al) for Al in ψ) do Aˡ
        @tullio M[a¹,b¹,aˡ⁺¹,bˡ,xˡ] := L[a¹,aˡ,b¹,bˡ] * conj(Aˡ[aˡ,aˡ⁺¹,xˡ])
        @tullio L[a¹,aˡ⁺¹,b¹,bˡ⁺¹] := M[a¹,b¹,aˡ⁺¹,bˡ,xˡ] * Aˡ[bˡ,bˡ⁺¹,xˡ]
        @debug @assert L ≈ conj(permutedims(L, (3,4,1,2)))
        # restore hermiticity after possible numerical errors
        L .= (conj(permutedims(L, (3,4,1,2))) + L) / 2
    end
end

function TensorTrains.accumulate_R(p::MatrixProductState)
    (; ψ) = p
    d = size(ψ[end], 2)
    R = id4(d)
    return map(_reshape1(Al) for Al in Iterators.reverse(ψ)) do Aˡ
        @tullio M[bˡ⁺¹,aᴸ,bᴸ,aˡ,xˡ] := R[aˡ⁺¹,aᴸ,bˡ⁺¹,bᴸ] * conj(Aˡ[aˡ,aˡ⁺¹,xˡ])
        @tullio R[aˡ,aᴸ,bˡ,bᴸ] := M[bˡ⁺¹,aᴸ,bᴸ,aˡ,xˡ] * Aˡ[bˡ,bˡ⁺¹,xˡ]
        # restore hermiticity after possible numerical errors
        @debug @assert R ≈ conj(permutedims(R, (3,4,1,2)))
        R .= (conj(permutedims(R, (3,4,1,2))) + R) / 2
    end |> reverse
end

function trace(A::Array{T,4}) where T
    @tullio t = A[a,a,b,b]
end

function TensorTrains.normalization(p::MatrixProductState; l = accumulate_L(p))
    lᴸ = l[end]
    @tullio z = lᴸ[a,a,b,b]
    @debug let r = accumulate_R(p)
        zr = trace(rfirst(r))
        @assert zr ≈ z "z=$z, got $zr, p=$p"  # sanity check
    end
    z
end

function TensorTrains.normalize!(p::MatrixProductState)
    Z = normalization(p)
    L = length(p)
    for a in p
        a ./= Z^(1/2L)
    end
    log(Z)
end

function TensorTrains.marginals(p::MatrixProductState;
    l = accumulate_L(p), r = accumulate_R(p))
    (; ψ) = p
    d = size(ψ[begin], 1)

    map(eachindex(ψ)) do t 
        Aᵗ = _reshape1(ψ[t])
        R = t + 1 ≤ length(ψ) ? r[t+1] : id4(d)
        L = t - 1 ≥ 1 ? l[t-1] : id4(d)
        @tullio lA[a¹,aᵗ⁺¹,b¹,bᵗ,x] := L[a¹,aᵗ,b¹,bᵗ] * conj(Aᵗ[aᵗ,aᵗ⁺¹,x])
        @tullio rA[aᵗ⁺¹,a¹,bᵗ,b¹,x] := Aᵗ[bᵗ,bᵗ⁺¹,x] * R[aᵗ⁺¹,a¹,bᵗ⁺¹,b¹]
        @tullio pᵗ[x] := lA[a¹,aᵗ⁺¹,b¹,bᵗ,x] * rA[aᵗ⁺¹,a¹,bᵗ,b¹,x]
        pᵗ ./= sum(pᵗ)
        @debug @assert real(pᵗ) ≈ pᵗ
        pᵗ = real(pᵗ)  
        reshape(pᵗ, size(ψ[t])[3:end])
    end
end

function TensorTrains.orthogonalize_right!(p::MatrixProductState; svd_trunc=TruncThresh(1e-6))
    orthogonalize_right!(p.ψ; svd_trunc)
end

function TensorTrains.orthogonalize_left!(p::MatrixProductState; svd_trunc=TruncThresh(1e-6))
    orthogonalize_left!(p.ψ; svd_trunc)
end

function TensorTrains.compress!(p::MatrixProductState; svd_trunc=TruncThresh(1e-6))
    compress!(p.ψ; svd_trunc)
end

"""
    sample!([rng], x, p::MatrixProductState; r)

Draw an exact sample from `p` and store the result in `x`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(p)`.

The output is `x,q`, the sampled sequence and its probability
"""
function TensorTrains.sample!(rng::AbstractRNG, x, p::MatrixProductState{<:AbstractTensorTrain{F,N}};
        r = accumulate_R(p)) where {F<:Number,N}
    L = length(p)
    @assert length(x) == L
    @assert all(length(xᵗ) == N-2 for xᵗ in x)
    (; ψ) = p
    d = size(ψ[end], 2)
    
    Q = Matrix(I, d, d)     # stores product of the first `l` matrices, evaluated at the sampled `x¹,...,xᵗ`
    for l in eachindex(p)
        rˡ⁺¹ = l == L ? [a==aᴸ * b == bᴸ for a in 1:d, aᴸ in 1:d, b in 1:d, bᴸ in 1:d] : r[l+1]
        # collapse multivariate xᵗ into 1D vector, sample from it
        Aˡ = _reshape1(ψ[l])
        @tullio QA[k,n,x] := Q[k,m] * Aˡ[m,n,x]
        @tullio q[x] := conj(QA[a¹,aˡ⁺¹,x]) * rˡ⁺¹[aˡ⁺¹,a¹,bˡ⁺¹,b¹] * QA[b¹,bˡ⁺¹,x]
        q ./= sum(q)
        @debug @assert q ≈ real(q)
        q = real(q)
        xˡ = sample_noalloc(rng, q)
        x[l] .= CartesianIndices(size(ψ[l])[3:end])[xˡ] |> Tuple
        # update prob
        Q = QA[:,:,xˡ]
    end
    q = abs2(tr(Q)) / trace(first(r))
    return x, q
end

"""
    sample([rng], p::MatrixProductState; r)

Draw an exact sample from `p`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(p)`.

The output is `x,q`, the sampled sequence and its probability
"""
function StatsBase.sample(rng::AbstractRNG, p::MatrixProductState{<:AbstractTensorTrain{F,N}};
        r = accumulate_R(p)) where {F<:Number,N}
    x = [zeros(Int, N-2) for Aᵗ in p]
    sample!(rng, x, p; r)
end
function StatsBase.sample(p::MatrixProductState; r = accumulate_R(p))
    sample(default_rng(), p; r)
end

function StatsBase.sample!(x, p::MatrixProductState; r = accumulate_R(p))
    sample!(default_rng(), x, p; r)
end

end # module