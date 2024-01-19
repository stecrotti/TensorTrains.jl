"""
    AbstractTensorTrain

An abstract type representing a Tensor Train.
It currently supports 2 subtypes [`TensorTrain`](@ref) and [`PeriodicTensorTrain`](@ref).
"""
abstract type AbstractTensorTrain{F<:Number, N} end

Base.eltype(::AbstractTensorTrain{F,N}) where {N,F} = F


"""
    bond_dims(A::AbstractTensorTrain)

Return a vector with the dimensions of the virtual bonds
"""
bond_dims(A::AbstractTensorTrain) = [size(A[t], 1) for t in 1:lastindex(A)]

function check_bond_dims(tensors::Vector{<:Array})
    for t in 1:lastindex(tensors)
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
evaluate(A::AbstractTensorTrain, X...) = tr(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))

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

    M = fill(zeros(0, 0), L, L)

    for u in 2:L
        Au = trace(A[u-1])
        for t in 1:u-2
            M[t, u] = M[t, u-1] * Au
        end
        # initial condition
        M[u-1, u] = Matrix(I, size(A[u],1), size(A[u],1))
    end

    return M
end

trace(At) = @tullio _[aᵗ,aᵗ⁺¹] := _reshape1(At)[aᵗ,aᵗ⁺¹,x]

function accumulate_L(A::AbstractTensorTrain)
    L = Matrix(I, size(A[begin],1), size(A[begin],1))
    map(trace(Atx) for Atx in A) do At
        L = L * At
    end
end

function accumulate_R(A::AbstractTensorTrain)
    R = Matrix(I, size(A[end],2), size(A[end],2))
    map(trace(Atx) for Atx in Iterators.reverse(A)) do At
        R = At * R
    end |> reverse
end

"""
    marginals(A::AbstractTensorTrain; l, r)

Compute the marginal distributions ``p(x^l)`` at each site

### Optional arguments
- `l = accumulate_L(A)`, `r = accumulate_R(A)` pre-computed partial normalizations
"""
function marginals(A::AbstractTensorTrain{F,N};
    l = accumulate_L(A), r = accumulate_R(A)) where {F<:Real,N}

    map(eachindex(A)) do t 
        Aᵗ = _reshape1(A[t])
        R = t + 1 ≤ length(A) ? r[t+1] : Matrix(I, size(A[end],2), size(A[end],2))
        L = t - 1 ≥ 1 ? l[t-1] : Matrix(I, size(A[begin],1), size(A[begin],1))
        @tullio lA[a¹,aᵗ⁺¹,x] := L[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x]
        @tullio pᵗ[x] := lA[a¹,aᵗ⁺¹,x] * R[aᵗ⁺¹,a¹]
        #@reduce pᵗ[x] := sum(a¹,aᵗ,aᵗ⁺¹) lᵗ⁻¹[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * rᵗ⁺¹[aᵗ⁺¹,a¹]  
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end
end

"""
    twovar_marginals(A::AbstractTensorTrain; l, r, M, Δlmax)

Compute the marginal distributions for each pair of sites ``p(x^l, x^m)``

### Optional arguments
- `l = accumulate_L(A)`, `r = accumulate_R(A)`, `M = accumulate_M(A)` pre-computed partial normalizations
- `maxdist = length(A)`: compute marginals only at distance `maxdist`: ``|l-m|\\le maxdist``
"""
function twovar_marginals(A::AbstractTensorTrain{F,N};
    l = accumulate_L(A), r = accumulate_R(A), M = accumulate_M(A),
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

            #@tullio bᵗᵘ[xᵗ, xᵘ] :=
            #lᵗ⁻¹[a¹,aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ] * 
            #Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * rᵘ⁺¹[aᵘ⁺¹,a¹]

            bᵗᵘ ./= sum(bᵗᵘ)
            b[t,u] = reshape(bᵗᵘ, qs)
        end
    end
    b
end

"""
    normalization(A::AbstractTensorTrain; l, r)

Compute the normalization ``Z=\\sum_{x^1,\\ldots,x^L} A^1(x^1)\\cdots A^L(x^L)``
"""
function normalization(A::AbstractTensorTrain; l = accumulate_L(A))
    z = tr(l[end])
    @debug let r = accumulate_R(A)
        @assert tr(r[begin]) ≈ z "z=$z, got $(tr(r[begin])), A=$A"  # sanity check
    end
    z
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
    sample!([rng], x, A::AbstractTensorTrain; r)

Draw an exact sample from `A` and store the result in `x`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function StatsBase.sample!(rng::AbstractRNG, x, A::AbstractTensorTrain{F,N};
        r = accumulate_R(A)) where {F<:Real,N}
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
        p ./= sum(p)
        xᵗ = sample_noalloc(rng, p)
        x[t] .= CartesianIndices(size(A[t])[3:end])[xᵗ] |> Tuple
        # update prob
        Q = Q * Aᵗ[:,:,xᵗ]
    end
    p = tr(Q) / tr(first(r))
    return x, p
    end

"""
    sample([rng], A::AbstractTensorTrain; r)

Draw an exact sample from `A`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.default_rng()`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function StatsBase.sample(rng::AbstractRNG, A::AbstractTensorTrain{F,N};
        r = accumulate_R(A)) where {F<:Real,N}
    x = [zeros(Int, N-2) for Aᵗ in A]
    sample!(rng, x, A; r)
end
function StatsBase.sample(A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample(default_rng(), A; r)
end

function StatsBase.sample!(x, A::AbstractTensorTrain{F,N}; r = accumulate_R(A)) where {F<:Real,N}
    sample!(default_rng(), x, A; r)
end

@doc raw"""
    LinearAlgebra.dot(ψ::AbstractTensorTrain, ϕ::AbstractTensorTrain)

Compute the inner product between tensor trains ``\psi(x^1,x^2,\ldots,x^L)=A^1(x^1)A^2(x^2)\cdots A^L(x^L)`` and ``\phi(x^1,x^2,\ldots,x^L)=B^1(x^1)B^2(x^2)\cdots B^L(x^L)``

```math
\psi\cdot \phi = \sum_{x^1,x^2,\ldots,x^L} \psi^*(x^1,x^2,\ldots,x^L) \phi(x^1,x^2,\ldots,x^L)
```
where ``^*`` stands for complex conjugate.
"""
function LinearAlgebra.dot(ψ::AbstractTensorTrain, ϕ::AbstractTensorTrain)
    ψᴸ = _reshape1(ψ[end])
    ϕᴸ = _reshape1(ϕ[end])
    @tullio C[aᴸ,a¹,b¹,bᴸ] := conj(ψᴸ[aᴸ,a¹,xᴸ]) * ϕᴸ[bᴸ,b¹,xᴸ]

    for (ψl, ϕl) in Iterators.drop(Iterators.reverse(zip(ψ,ϕ)), 1)
        ψˡ = _reshape1(ψl)
        ϕˡ = _reshape1(ϕl)
        @tullio Cnew[aˡ,a¹,b¹,bˡ] := conj(ψˡ[aˡ,aˡ⁺¹,xˡ]) * C[aˡ⁺¹,a¹,b¹,bˡ⁺¹] * ϕˡ[bˡ,bˡ⁺¹,xˡ]
        C = Cnew
    end

    @tullio d = C[a¹,a¹,b¹,b¹]
end

@doc raw"""
    LinearAlgebra.norm(ψ::AbstractTensorTrain)

Compute the 2-norm (Frobenius norm) of tensor train `ψ`

```math
\lVert ψ\rVert_2 = \sqrt{ψ\cdot ψ}
```
"""
LinearAlgebra.norm(ψ::AbstractTensorTrain) = sqrt(dot(ψ, ψ))

@doc raw"""
    norm2m(ψ::AbstractTensorTrain, ϕ::AbstractTensorTrain)

Given two tensor trains `ψ,ϕ`, compute `norm(ψ - ϕ)^2` as

```math
\lVert ψ-ϕ\rVert_2^2 = \lVert ψ \rVert_2^2 + \lVert ϕ \rVert_2^2 - 2ψ\cdot ϕ
```
"""
function norm2m(ψ::AbstractTensorTrain, ϕ::AbstractTensorTrain) 
    return norm(ψ)^2 + norm(ϕ)^2 - 2*dot(ψ, ϕ)
end

"""
    LinearAlgebra.normalize!(p::AbstractTensorTrain)

Normalize `p` to a probability distribution
"""
function LinearAlgebra.normalize!(p::AbstractTensorTrain)
    c = normalize_eachmatrix!(p)
    Z = normalization(p)
    L = length(p)
    for a in p
        a ./= Z^(1/L)
    end
    c + log(Z)
end
