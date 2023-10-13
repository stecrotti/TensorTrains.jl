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
    @tullio p[x] := Q[k,m] * Aᵗ[m,n,x] * rᵗ⁺¹[n,k]
    p ./= sum(p)
    xᵗ = sample_noalloc(rng, p)
    x[t] .= CartesianIndices(size(A[t])[3:end])[xᵗ] |> Tuple
    # update prob
    Q = Q * Aᵗ[:,:,xᵗ]
end
p = tr(Q) / tr(first(r))
return x, p
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

    A¹ = _reshape1(A[begin]); r² = r[2]
    @tullio p¹[x] := A¹[a¹,a²,x] * r²[a²,a¹]
    p¹ ./= sum(p¹)
    p¹ = reshape(p¹, size(A[begin])[3:end])

    Aᴸ = _reshape1(A[end]); lᴸ⁻¹ = l[end-1]
    @tullio pᴸ[x] := lᴸ⁻¹[a¹,aᴸ] * Aᴸ[aᴸ,a¹,x]
    pᴸ ./= sum(pᴸ)
    pᴸ = reshape(pᴸ, size(A[end])[3:end])

    p = map(2:length(A)-1) do t 
        Aᵗ = _reshape1(A[t])
        rl = r[t+1] * l[t-1]
        @tullio pᵗ[x] := rl[aᵗ⁺¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x]  
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end

    return append!([p¹], p, [pᴸ])
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
            @tullio bᵗᵘ[xᵗ, xᵘ] :=
                lᵗ⁻¹[a¹,aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * rᵘ⁺¹[aᵘ⁺¹,a¹]
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
function normalization(A::AbstractTensorTrain; l = accumulate_L(A), r = accumulate_R(A))
    z = tr(l[end])
    @assert tr(r[begin]) ≈ z "z=$z, got $(tr(r[begin])), A=$A"  # sanity check
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