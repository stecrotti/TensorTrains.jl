
"""
    MPS{T<:AbstractTensorTrain}

With a little abuse of nomenclature, a type for representing a probability distribution whose value is the square module of the value of the contained tensor train.

# FIELDS
- `ψ`: an `AbstractTensorTrain`

Example:
```@example
    L = 3
    q = (2, 3)
    ψ = rand_mps(4, L, q...)
    p = MPS(ψ)
    X = [[rand(1:qi) for qi in q] for l in 1:L]
    evaluate(p, X), abs2(evaluate(ψ, X))    # are the same
```
"""
struct MPS{T<:AbstractTensorTrain}
    ψ :: T
end

@forward MPS.ψ TensorTrains.bond_dims, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.getindex, check_bond_dims, Base.length, Base.eachindex,
    TensorTrains.nparams

Base.:(==)(A::T, B::T) where {T<:MPS} = isequal(A.ψ, B.ψ)
Base.isapprox(A::T, B::T; kw...) where {T<:MPS} = isapprox(A.ψ, B.ψ; kw...)
TensorTrains.is_in_domain(p::MPS, X...) = is_in_domain(p.ψ, X...)

"""
    rand_mps([T = Float64], bondsizes::AbstractVector{<:Integer}, q...)
    rand_mps([T = Float64], d::Integer, L::Integer, q...)

Construct a Matrix Product States with `rand(T)` entries, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_mps(t::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where T <: Number
    return MPS(rand_tt(t, bondsizes, q...))
end
rand_mps(bondsizes::AbstractVector{<:Integer}, q...) = rand_mps(Float64, bondsizes, q...)
rand_mps(::Type{T}, d::Integer, L::Integer, q...) where {T <: Number} = rand_mps(T, [1; fill(d, L-1); 1], q...)
rand_mps(d::Integer, L::Integer, q...) = rand_mps(Float64, d, L, q...)


TensorTrains.is_left_canonical(A::MPS; kw...) = is_left_canonical(A.ψ; kw...)
TensorTrains.is_right_canonical(A::MPS; kw...) = is_right_canonical(A.ψ; kw...)
TensorTrains.is_canonical(A::MPS; kw...) = is_canonical(A.ψ; kw...)

"""
    evaluate(p::PMS, X...)

Return the value of `p` for input `X`.
If `ψ` is the tensor train wrapped by `p`, then the output is ``\\lvert\\psi (X)\\rvert^2``
"""
TensorTrains.evaluate(p::MPS, X...) = abs2(evaluate(p.ψ, X...))

id4(d::Integer) = [a==a¹ && b==b¹ for a in 1:d, a¹ in 1:d, b in 1:d, b¹ in 1:d]

function TensorTrains.accumulate_L(p::MPS{<:AbstractTensorTrain{F}}; normalize=true) where {F}
    (; ψ) = p
    d = size(ψ[begin], 1)
    Ll = id4(d)
    z = Logarithmic(one(F))
    L = map(_reshape1(Al) for Al in ψ) do Aˡ
        nl = maximum(abs, Ll)
        if !iszero(nl) && normalize
            Ll ./= nl
            z *= nl
        end
        @tullio M[a¹,b¹,aˡ⁺¹,bˡ,xˡ] := Ll[a¹,aˡ,b¹,bˡ] * conj(Aˡ[aˡ,aˡ⁺¹,xˡ])
        @tullio Ll[a¹,aˡ⁺¹,b¹,bˡ⁺¹] := M[a¹,b¹,aˡ⁺¹,bˡ,xˡ] * Aˡ[bˡ,bˡ⁺¹,xˡ]
        # restore hermiticity after possible numerical errors
        @debug @assert Ll ≈ conj(permutedims(Ll, (3,4,1,2)))
        Ll .= (conj(permutedims(Ll, (3,4,1,2))) + Ll) / 2
    end
    z *= trace(Ll)
    @debug @assert real(z) ≈ z
    return L, real(z)
end

function TensorTrains.accumulate_R(p::MPS{<:AbstractTensorTrain{F}}; normalize=true) where {F}
    (; ψ) = p
    d = size(ψ[end], 2)
    Rl = id4(d)
    z = Logarithmic(one(F))
    R = map(_reshape1(Al) for Al in Iterators.reverse(ψ)) do Aˡ
        nl = maximum(abs, Rl)
        if !iszero(nl) && normalize
            Rl ./= nl
            z *= nl
        end
        @tullio M[bˡ⁺¹,aᴸ,bᴸ,aˡ,xˡ] := Rl[aˡ⁺¹,aᴸ,bˡ⁺¹,bᴸ] * conj(Aˡ[aˡ,aˡ⁺¹,xˡ])
        @tullio Rl[aˡ,aᴸ,bˡ,bᴸ] := M[bˡ⁺¹,aᴸ,bᴸ,aˡ,xˡ] * Aˡ[bˡ,bˡ⁺¹,xˡ]
        # restore hermiticity after possible numerical errors
        @debug @assert Rl ≈ conj(permutedims(Rl, (3,4,1,2)))
        Rl .= (conj(permutedims(Rl, (3,4,1,2))) + Rl) / 2
    end |> reverse
    z *= trace(Rl)
    @debug @assert real(z) ≈ z
    return R, real(z)
end

function trace(A::Array{T,4}) where T
    @tullio t = A[a,a,b,b]
end

# TODO: this is somehow a duplicate of norm(::TensorTrain)
"""
    normalization(p::MPS)

Compute the normalization ``Z=\\sum_{x^1,\\ldots,x^L} \\lvert A^1(x^1)\\cdots A^L(x^L)\\rvert ^2`` and return it as a `Logarithmic`, which stores the sign and the logarithm of the absolute value (see the docs of LogarithmicNumbers.jl https://github.com/cjdoris/LogarithmicNumbers.jl?tab=readme-ov-file#documentation)
"""
function TensorTrains.normalization(p::MPS; normalize_while_accumulating=true)
    l, z = accumulate_L(p; normalize=normalize_while_accumulating)
    @debug begin
        r, zr = accumulate_R(p; normalize=normalize_while_accumulating)
        @assert zr ≈ z "z=$z, got $zr, p=$p"  # sanity check
    end
    return z / abs2(p.ψ.z)
end

"""
    normalize!(p::MPS)

Compute the normalization of ``Z=\\sum_{x^1,\\ldots,x^L} \\lvert A^1(x^1)\\cdots A^L(x^L)\\rvert ^2`` (see [`normalization`](@ref)) and rescale the tensors in `p` such that, after this call, ``|Z|^2=1``.
Return the natural logarithm of the absolute normalization ``\\log|Z|``
"""
function TensorTrains.normalize!(p::MPS)
    Z = accumulate_L(p)[2]
    absZ = sqrt(abs2(Z))    # just abs fails for complex numbers (see https://github.com/cjdoris/LogarithmicNumbers.jl/issues/23)
    L = length(p)
    x = exp(1/L * log(sqrt(absZ)))
    if x != 0
        for a in p.ψ
            a ./= x
        end
    end
    logz = log(absZ / abs2(p.ψ.z))
    p.ψ.z = 1
    return logz
end

"""
    marginals(p::MPS; l, r)

Compute the marginal distributions ``p(x^l)`` at each site

### Optional arguments
- `l = accumulate_L(A)[1]`, `r = accumulate_R(A)[1]` pre-computed partial normalizations
"""
function TensorTrains.marginals(p::MPS;
    l = accumulate_L(p)[1], r = accumulate_R(p)[1])
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

function TensorTrains.orthogonalize_right!(p::MPS; kw...)
    orthogonalize_right!(p.ψ; kw...)
end

function TensorTrains.orthogonalize_left!(p::MPS; kw...)
    orthogonalize_left!(p.ψ; kw...)
end

function TensorTrains.orthogonalize_center!(p::MPS, l::Integer; kw...)
    orthogonalize_center!(p.ψ, l; kw...)
end

function TensorTrains.orthogonalize_two_site_center!(p::MPS, l::Integer; kw...)
    orthogonalize_two_site_center!(p.ψ, l; kw...)
end

function TensorTrains.compress!(p::MPS; kw...)
    compress!(p.ψ; kw...)
end

"""
    sample!([rng], x, p::MPS; r)

Draw an exact sample from `p` and store the result in `x`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(p)`.

The output is `x,q`, the sampled sequence and its probability
"""
function TensorTrains.sample!(rng::AbstractRNG, x, p::MPS{<:AbstractTensorTrain{F,N}};
        rz = accumulate_R(p)) where {F<:Number,N}
    r, z = rz
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
    q = abs2(tr(Q)) / z #trace(first(r))
    return x, q
end

"""
    sample([rng], p::MPS; r)

Draw an exact sample from `p`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(p)`.

The output is `x,q`, the sampled sequence and its probability
"""
function StatsBase.sample(rng::AbstractRNG, p::MPS{<:AbstractTensorTrain{F,N}};
        rz = accumulate_R(p)) where {F<:Number,N}
    x = [zeros(Int, N-2) for Aᵗ in p]
    sample!(rng, x, p; rz)
end
function StatsBase.sample(p::MPS; rz = accumulate_R(p))
    sample(default_rng(), p; rz)
end

function StatsBase.sample!(x, p::MPS; rz = accumulate_R(p))
    sample!(default_rng(), x, p; rz)
end


# TODO: maybe since (p.ψ.z) is both at the numerator and denominator, ignore it to avoid cancellations with the subtraction?

"""
    loglikelihood(p::MPS, X)

Compute the loglikelihood of the data `X` under the MPS distribution `p`.
"""
function loglikelihood(p::MPS, X)
    logz = log(normalization(p))
    return mean(log(evaluate(p, x)) for x in X) - logz 
end