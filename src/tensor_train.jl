"""
    TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
struct TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Array{F,N}}

    function TensorTrain(tensors::Vector{Array{F,N}}) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors)
    end
end


@forward TensorTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex


function check_bond_dims(tensors::Vector{<:Array})
    for t in 1:lastindex(tensors)-1
        dᵗ = size(tensors[t],2)
        dᵗ⁺¹ = size(tensors[t+1],1)
        if dᵗ != dᵗ⁺¹
            println("Bond size for matrix t=$t. dᵗ=$dᵗ, dᵗ⁺¹=$dᵗ⁺¹")
            return false
        end
    end
    return true
end
  
"""
    uniform_tt(bondsizes::AbstractVector{<:Integer}, q...)
    uniform_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train full of 1's, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function uniform_tt(bondsizes::AbstractVector{<:Integer}, q...)
    TensorTrain([ones(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end
uniform_tt(d::Integer, L::Integer, q...) = uniform_tt([1; fill(d, L-1); 1], q...)

"""
    rand_tt(bondsizes::AbstractVector{<:Integer}, q...)
    rand_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train with entries random in [0,1], by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_tt(bondsizes::AbstractVector{<:Integer}, q...)
    TensorTrain([rand(bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end
rand_tt(d::Integer, L::Integer, q...) = rand_tt([1; fill(d, L-1); 1], q...)

"""
    bond_dims(A::AbstractTensorTrain)

Return a vector with the dimensions of the virtual bonds
"""
bond_dims(A::TensorTrain) = [size(A[t], 2) for t in 1:lastindex(A)-1]


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
evaluate(A::TensorTrain, X...) = only(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))


"""
    orthogonalize_right!(A::TensorTrain; svd_trunc::SVDTrunc)

Bring `A` to right-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_right!(C::TensorTrain; svd_trunc=TruncThresh(1e-6))
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])
    return C
end

"""
    orthogonalize_left!(A::TensorTrain; svd_trunc::SVDTrunc)

Bring `A` to left-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
"""
function orthogonalize_left!(C::TensorTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    D = fill(1.0,1,1,1)

    for t in 1:length(C)-1
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x in 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    C[end] = _reshapeas(D, C[end])
    return C
end

"""
    compress!(A::TensorTrain; svd_trunc::SVDTrunc)

Compress `A` by means of SVD decompositions + truncations
"""
function compress!(A::TensorTrain; svd_trunc=TruncThresh(1e-6))
    orthogonalize_right!(A, svd_trunc=TruncThresh(0.0))
    orthogonalize_left!(A; svd_trunc)
end

function accumulate_L(A::TensorTrain)
    l = [zeros(0) for _ in eachindex(A)]
    A⁰ = _reshape1(A[begin])
    @reduce l⁰[a¹] := sum(x) A⁰[1,a¹,x]
    l[1] = l⁰

    lᵗ = l⁰
    for t in 1:length(A)-1
        Aᵗ = _reshape1(A[t+1])
        @reduce lᵗ[aᵗ⁺¹] |= sum(x,aᵗ) lᵗ[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] 
        l[t+1] = lᵗ
    end
    return l
end

function accumulate_R(A::TensorTrain)
    r = [zeros(0) for _ in eachindex(A)]
    Aᵀ = _reshape1(A[end])
    @reduce rᵀ[aᵀ] := sum(x) Aᵀ[aᵀ,1,x]
    r[end] = rᵀ

    rᵗ = rᵀ
    for t in length(A)-1:-1:1
        Aᵗ = _reshape1(A[t])
        @reduce rᵗ[aᵗ] |= sum(x,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,x] * rᵗ[aᵗ⁺¹] 
        r[t] = rᵗ
    end
    return r
end

"""
    marginals(A::AbstractTensorTrain; l, r)

Compute the marginal distributions ``p(x^l)`` at each site

### Optional arguments
- `l = accumulate_L(A)`, `r = accumulate_R(A)` pre-computed partial nommalizations
"""
function marginals(A::TensorTrain{F,N};
        l = accumulate_L(A), r = accumulate_R(A)) where {F<:Real,N}
    
    A⁰ = _reshape1(A[begin]); r¹ = r[2]
    @reduce p⁰[x] := sum(a¹) A⁰[1,a¹,x] * r¹[a¹]
    p⁰ ./= sum(p⁰)
    p⁰ = reshape(p⁰, size(A[begin])[3:end])

    Aᵀ = _reshape1(A[end]); lᵀ⁻¹ = l[end-1]
    @reduce pᵀ[x] := sum(aᵀ) lᵀ⁻¹[aᵀ] * Aᵀ[aᵀ,1,x]
    pᵀ ./= sum(pᵀ)
    pᵀ = reshape(pᵀ, size(A[end])[3:end])

    p = map(2:length(A)-1) do t 
        lᵗ⁻¹ = l[t-1]
        Aᵗ = _reshape1(A[t])
        rᵗ⁺¹ = r[t+1]
        @reduce pᵗ[x] := sum(aᵗ,aᵗ⁺¹) lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * rᵗ⁺¹[aᵗ⁺¹]  
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end

    return append!([p⁰], p, [pᵀ])
end

"""
    twovar_marginals(A::AbstractTensorTrain; l, r, M, Δlmax)

Compute the marginal distributions for each pair of sites ``p(x^l, x^m)``

### Optional arguments
- `l = accumulate_L(A)`, `r = accumulate_R(A)`, `M = accumulate_M(A)` pre-computed partial normalizations
- `maxdist = length(A)`: compute marginals only at distance `maxdist`: ``|l-m|\\le maxdist``
"""
function twovar_marginals(A::TensorTrain{F,N};
        l = accumulate_L(A), r = accumulate_R(A), M = accumulate_M(A),
        maxdist = length(A)-1) where {F<:Real,N}
    qs = tuple(reduce(vcat, [x,x] for x in size(A[begin])[3:end])...)
    b = Array{F,2*(N-2)}[zeros(zeros(Int, 2*(N-2))...) 
        for _ in eachindex(A), _ in eachindex(A)]
    for t in 1:length(A)-1
        lᵗ⁻¹ = t == 1 ? [1.0;] : l[t-1]
        Aᵗ = _reshape1(A[t])
        for u in t+1:min(length(A),t+maxdist)
            rᵘ⁺¹ = u == length(A) ? [1.0;] : r[u+1]
            Aᵘ = _reshape1(A[u])
            Mᵗᵘ = M[t, u]
            @tullio bᵗᵘ[xᵗ, xᵘ] :=
                lᵗ⁻¹[aᵗ] * Aᵗ[aᵗ, aᵗ⁺¹, xᵗ] * Mᵗᵘ[aᵗ⁺¹, aᵘ] * 
                Aᵘ[aᵘ, aᵘ⁺¹, xᵘ] * rᵘ⁺¹[aᵘ⁺¹]
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
function normalization(A::TensorTrain; l = accumulate_L(A), r = accumulate_R(A))
    z = only(l[end])
    @assert only(r[begin]) ≈ z "z=$z, got $(only(r[begin])), A=$A"  # sanity check
    z
end

"""
    normalize!(A::AbstractTensorTrain)

Normalize `A` to a probability distribution
"""
function normalize!(A::AbstractTensorTrain)
    c = normalize_eachmatrix!(A)
    Z = normalization(A)
    L = length(A)
    for a in A
        a ./= Z^(1/L)
    end
    c + log(Z)
end

# used to do stuff like `A+B` with `A,B` tensor trains
function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB}) where {F,NA,NB}
    @assert NA == NB
    @assert length(A) == length(B)
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ hcat(Aᵗ[:,:,x...], f(Bᵗ[:,:,x...])) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), 1, sa[2]+sb[2], size(Aᵗ)[3:end]...)
        elseif t == lastindex(A)
            Cᵗ = [ vcat(Aᵗ[:,:,x...], Bᵗ[:,:,x...]) 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), sa[1]+sb[1], 1, size(Aᵗ)[3:end]...)
        else
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bᵗ[:,:,x...]] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        end
    end
    TensorTrain(tensors)
end

"""
    sample!([rng], x, A::AbstractTensorTrain; r)

Draw an exact sample from `A` and store the result in `x`.

Optionally specify a random number generator `rng` as the first argument
  (defaults to `Random.GLOBAL_RNG`) and provide a pre-computed `r = accumulate_R(A)`.

The output is `x,p`, the sampled sequence and its probability
"""
function sample!(rng::AbstractRNG, x, A::TensorTrain{F,N};
        r = accumulate_R(A)) where {F<:Real,N}
    L = length(A)
    @assert length(x) == L
    @assert all(length(xᵗ) == N-2 for xᵗ in x)

    Q = ones(F, 1, 1)  # stores product of the first `t` matrices, evaluated at the sampled `x¹,...,xᵗ`
    for t in eachindex(A)
        rᵗ⁺¹ = t == L ? ones(F,1) : r[t+1]
        # collapse multivariate xᵗ into 1D vector, sample from it
        Aᵗ = _reshape1(A[t])
        @tullio p[x] := Q[m] * Aᵗ[m,n,x] * rᵗ⁺¹[n]
        p ./= sum(p)
        xᵗ = sample_noalloc(rng, p)
        x[t] .= CartesianIndices(size(A[t])[3:end])[xᵗ] |> Tuple
        # update prob
        Q = Q * Aᵗ[:,:,xᵗ]
    end
    p = only(Q) / only(first(r))
    return x, p
end