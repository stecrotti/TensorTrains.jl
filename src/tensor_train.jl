"""
    TensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
mutable struct TensorTrain{F<:Number, N, T, Z} <: AbstractTensorTrain{F,N}
    tensors::Vector{T}
    z::Z

    function TensorTrain{F,N}(tensors::Vector{T}; z::Z=Logarithmic(one(F))) where {F<:Number, N, T <: AbstractArray{F,N}, Z}
        N > 2 || throw(ArgumentError("Tensors should have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) == 1 ||
            throw(ArgumentError("First matrix must have 1 row, last matrix must have 1 column"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N,T,Z}(tensors, z)
    end
    TensorTrain{F,N,T,Z}(tensors; z::Z=Logarithmic(one(F))) where {F,N,T,Z} = TensorTrain{F,N}(tensors; z)
end
function TensorTrain(tensors::Vector{<:AbstractArray{F,N}}; z=one(F)) where {F<:Number, N}
    return TensorTrain{F,N}(tensors; z = Logarithmic(z))
end


@forward TensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex, Base.reverse
    check_bond_dims


"""
    flat_tt([T = Float64], bondsizes::AbstractVector{<:Integer}, q...)
    flat_tt([T = Float64], d::Integer, L::Integer, q...)

Construct a (normalized) Tensor Train filled with `one(T)`, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function flat_tt(::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where T<:Number
    TensorTrain([fill(one(T), bondsizes[t], bondsizes[t+1], q...) for t in 1:length(bondsizes)-1])
end

flat_tt(bondsizes::AbstractVector{<:Integer}, q...) = flat_tt(Float64, bondsizes, q...)

flat_tt(d::Integer, L::Integer, q...) = flat_tt([1; fill(d, L-1); 1], q...)


"""
    rand_tt([rng=default_rng()], [T = Float64], bondsizes::AbstractVector{<:Integer}, q...)
    rand_tt([rng=default_rng()], [T = Float64], d::Integer, L::Integer, q...)

Construct a Tensor Train with `rand(T)` entries, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_tt(rng::AbstractRNG, ::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where T <: Number
    A = flat_tt(T, bondsizes, q...)
    foreach(a->(a .= rand.(rng, T)), A)
    return A
end

function rand_tt(rng::AbstractRNG, ::Type{T}, d::Integer, L::Integer, q...) where T <: Number
    return rand_tt(rng, T, [1; fill(d, L-1); 1], q...)
end

rand_tt(rng::AbstractRNG, bondsizes::AbstractVector{<:Integer}, q...) = rand_tt(rng, Float64, bondsizes, q...)
rand_tt(::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where {T <: Number} = rand_tt(default_rng(), T, bondsizes, q...)
rand_tt(bondsizes::AbstractVector{<:Integer}, q...) = rand_tt(default_rng(), Float64, bondsizes, q...)

rand_tt(rng::AbstractRNG, d::Integer, L::Integer, q...) = rand_tt(rng, Float64, [1; fill(d, L-1); 1], q...)
rand_tt(::Type{T}, d::Integer, L::Integer, q...) where {T <: Number} = rand_tt(default_rng(), T, [1; fill(d, L-1); 1], q...)
rand_tt(d::Integer, L::Integer, q...) = rand_tt(default_rng(), Float64, d, L, q...)


"""
    randn_tt([rng=default_rng()], [T = Float64], bondsizes::AbstractVector{<:Integer}, q...)
    randn_tt([rng=default_rng()], [T = Float64], d::Integer, L::Integer, q...)

Construct a Tensor Train with `randn(T)` entries, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function randn_tt(rng::AbstractRNG, ::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where T <: Number
    A = flat_tt(T, bondsizes, q...)
    d = maximum(bondsizes)
    σ = 1 / sqrt(d)
    foreach(a->(a .= σ .* randn.(rng, T)), A)
    return A
end

function randn_tt(rng::AbstractRNG, ::Type{T}, d::Integer, L::Integer, q...) where T <: Number
    return randn_tt(rng, T, [1; fill(d, L-1); 1], q...)
end

randn_tt(rng::AbstractRNG, bondsizes::AbstractVector{<:Integer}, q...) = randn_tt(rng, Float64, bondsizes, q...)
randn_tt(::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where {T <: Number} = randn_tt(default_rng(), T, bondsizes, q...)
randn_tt(bondsizes::AbstractVector{<:Integer}, q...) = randn_tt(default_rng(), Float64, bondsizes, q...)

randn_tt(rng::AbstractRNG, d::Integer, L::Integer, q...) = randn_tt(rng, Float64, [1; fill(d, L-1); 1], q...)
randn_tt(::Type{T}, d::Integer, L::Integer, q...) where {T <: Number} = randn_tt(default_rng(), T, [1; fill(d, L-1); 1], q...)
randn_tt(d::Integer, L::Integer, q...) = randn_tt(default_rng(), Float64, d, L, q...)

"""
    tune_scaling!(A::TensorTrain)

Heuristically re-scale the entries of `A` such that the expected RMS value of the output is about 1.

## Keyword arguments

- `ninputs = 10^2`: number of random inputs to use to estimate RMS value of output
- `target_stdev = 1.0`: target value for RMS output
- `grid = 10.0 .^ (-2:0.05:2)`: for gridsearch of best per-entry re-scaling factor 
"""
function tune_scaling!(A::TensorTrain;
    ninputs::Integer = 10^2, target_stdev::Real = 1.0, 
    grid = 10.0 .^ (-2:0.05:2), verbose::Bool = false)

    X = [[[rand(1:q) for q in size(Ai)[3:end]] for Ai in A] for _ in 1:ninputs]
    verbose && println("Tuning scaling...")
    stdevs = map(grid) do s
        B = deepcopy(A)
        for Bi in B
            Bi .*= s
        end
        std(evaluate.((B,), X))
    end
    stdev, i = findmin(filter!(!isnan, abs.(stdevs .- target_stdev)))
    s = grid[i]
    verbose && println("Selected entry-wise scaling factor $s which gave a RMS output value of $stdev")
    for Ai in A
        Ai .*= s
    end
    return A
end

"""
    orthogonalize_right!(A::AbstractTensorTrain; svd_trunc::SVDTrunc, indices)

Bring `A` to right-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
Optionally pass a range of indices to perform the orthogonalizations only on those.
"""
function orthogonalize_right!(C::TensorTrain{F}; svd_trunc=TruncThresh(1e-6),
    indices=2:lastindex(C)) where F

    isempty(indices) && return C
    issorted(indices) || throw(ArgumentError("Indices must be sorted"))
    all(id ∈ 2:lastindex(C) for id in extrema(indices)) || throw(ArgumentError("Indices not compatible with tensor train positions $(eachindex(C)): got $indices"))
    Cᵀ = _reshape1(C[end])
    q = size(Cᵀ, 3)
    @cast M[m, (n, x)] := Cᵀ[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(one(F))

    for t in Iterators.reverse(indices)
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[first(indices)-1] = _reshapeas(D, C[first(indices)-1])
    C.z /= c
    return C
end


"""
    orthogonalize_left!(A::AbstractTensorTrain; svd_trunc::SVDTrunc, indices::UnitRange)

Bring `A` to left-orthogonal form by means of SVD decompositions.

Optionally perform truncations by passing a `SVDTrunc`.
Optionally pass a range of indices to perform the orthogonalizations only on those.
"""
function orthogonalize_left!(C::TensorTrain{F}; svd_trunc=TruncThresh(1e-6),
    indices=1:lastindex(C)-1) where F

    isempty(indices) && return C
    issorted(indices) || throw(ArgumentError("Indices must be sorted"))
    all(id ∈ 1:lastindex(C)-1 for id in extrema(indices)) || throw(ArgumentError("Indices not compatible with tensor train positions $(eachindex(C)): got $indices"))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[(m, x), n] |= C⁰[m, n, x]
    D = fill(1.0,1,1,1)
    c = Logarithmic(one(F))

    for t in indices
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x ∈ 1:q
        C[t] = _reshapeas(Aᵗ, C[t])
        Cᵗ⁺¹ = _reshape1(C[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Cᵗ⁺¹[l, n, x]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[(m, x), n] |= D[m, n, x]
    end
    C[last(indices)+1] = _reshapeas(D, C[last(indices)+1])
    C.z /= c
    return C
end

function orthogonalize_center!(C::TensorTrain, l::Integer; svd_trunc=TruncThresh(1e-6))
    orthogonalize_left!(C; svd_trunc, indices = 1:l-1)
    orthogonalize_right!(C; svd_trunc, indices = l+1:length(C))
end

"""
    orthogonalize_two_site_center!(C::TensorTrain, k::Integer; svd_trunc=TruncThresh(1e-6))

Orthogonalize the tensor train for a two-site DMRG update at positions k and k+1.
This puts sites 1:k-1 in left-canonical form, sites k+2:N in right-canonical form,
and leaves sites k and k+1 as the non-orthogonal center for merging.
"""
function orthogonalize_two_site_center!(C::TensorTrain, k::Integer; svd_trunc=TruncThresh(1e-6))
    @assert 1 <= k < length(C) "k must be between 1 and length(C)-1 for two-site update"
    orthogonalize_left!(C; svd_trunc, indices = 1:k-1)
    orthogonalize_right!(C; svd_trunc, indices = k+2:length(C))
end


# used to do stuff like `A+B` with `A,B` tensor trains
function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB}) where {F,NA,NB}
    axes(A[1])[3:end] == axes(B[1])[3:end] || throw(ArgumentError("Tensor Trains must have the types of physical indices, got $(axes(A[1])[3:end]) and $(axes(B[1])[3:end])"))
    length(A) == length(B) || throw(ArgumentError("Tensor Trains must have the same length, got $(length(A)) and $(length(B))"))
    z = min(abs(A.z), abs(B.z))
    za, zb = float(z/A.z), f(float(z/B.z))
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        At, Bt = _reshape1(Aᵗ), _reshape1(Bᵗ)
        X = axes(At,3)
        @views if t == firstindex(A)
            Cᵗ = [[za*At[:,:,x] zb*Bt[:,:,x]] for x in X]
        elseif t == lastindex(A)
            Cᵗ = [[At[:,:,x]; Bt[:,:,x]] for x in X]
        else
            sa, sb = size(At),size(Bt)
            Cᵗ = [[At[:,:,x] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bt[:,:,x]] for x in X]
        end
        _reshapeas((@tullio _[i,j,x] := Cᵗ[x][i,j]), Aᵗ)
    end
    C = TensorTrain(tensors)
    C.z = z
    C
end

function is_left_canonical(A; atol=1e-10)
    A_resh = _reshape1(A)
    @tullio AA[i,j] := conj(A_resh[k,i,x]) * A_resh[k,j,x]
    return is_approx_identity(AA; atol)
end

function is_right_canonical(A; atol=1e-10)
    A_resh = _reshape1(A)
    @tullio AA[i,j] := A_resh[i,k,x] * conj(A_resh[j,k,x])
    return is_approx_identity(AA; atol)
end

function is_canonical(A, central_idx; atol=1e-10)
    f_l(x) = is_left_canonical(x; atol)
    f_r(x) = is_right_canonical(x; atol)
    return all(f_l, A[begin:begin+central_idx-2]) &&
        all(f_r, A[begin+central_idx:end])
end

"""
    is_two_site_canonical(A, k)

Check if the tensor train is in canonical form for a two-site update at positions k and k+1.
This means sites 1:k-1 are left-canonical, sites k+2:N are right-canonical,
and sites k and k+1 are non-canonical (ready for merging).
"""
function is_two_site_canonical(A, k; atol=1e-10)
    @assert 1 <= k < length(A) "k must be between 1 and length(A)-1 for two-site update"
    f_l(x) = is_left_canonical(x; atol)
    f_r(x) = is_right_canonical(x; atol)

    # Check left canonical: sites 1 to k-1
    left_canonical = k == 1 || all(f_l, A[1:k-1])

    # Check right canonical: sites k+2 to N
    right_canonical = k+1 == length(A) || all(f_r, A[k+2:end])

    return left_canonical && right_canonical
end

# TODO: return directly the grad of the log so the two z's will cancel out
# compute the gradient of evaluating the tensor train with respect to the entries of Aˡ
function grad_evaluate(A::TensorTrain, l::Integer, X)
    id = fill(one(eltype(A)), 1, 1)
    prodA_left = prod(@view a[:,:,x...] for (a,x) in zip(A[1:l-1], X[1:l-1]); init=id)
    prodA_right = reduce((A,B) -> B * A, @view a[:,:,x...] for (a,x) in zip(A[end:-1:l+1], X[end:-1:l+1]); init=id)
    Ax_center = A[l][:,:,X[l]...]
    z = float(A.z)
    val = only(prodA_left * Ax_center * prodA_right) / z
    gr = (prodA_right * prodA_left)' / z
    return gr, val
end

"
Pre-compute the product of all matrices to the left of k and to the right of k+1
for a single datapoint `X`
"
function precompute_left_environments(A::TensorTrain{F}, X) where {F}
    prodA_left = pushfirst!(
        accumulate((A,B) -> A * B, Ak[:,:,xk...] for (Ak,xk) in zip(A,X)),
        ones(F,1,1)
    )
    return OffsetArray(prodA_left, -1)
end

function precompute_right_environments(A::TensorTrain{F}, X) where F
    prodA_right = push!(
        reverse(accumulate((A,B) -> B * A, Ak[:,:,xk...] for (Ak,xk) in zip(reverse(A),reverse(X)))),
        ones(F,1,1)
    )
    return OffsetArray(prodA_right, 0)
end

# TODO: return directly the grad of the log so the two z's will cancel out
# compute the gradient of evaluating the tensor train with respect to the entries of Aˡ merged with Aˡ⁺¹
# TODO:
function grad_evaluate_two_site(A::TensorTrain, k::Integer, X;
    Ax_left = precompute_left_environments(A, X)[k-1],
    Ax_right = precompute_right_environments(A, X)[k+2],
    Aᵏᵏ⁺¹ = _merge_tensors(A[k], A[k+1])
)
    Ax_center = Aᵏᵏ⁺¹[:,:,X[k]...,X[k+1]...]
    z = float(A.z)
    val = only(Ax_left * Ax_center * Ax_right) / z
    gr = (Ax_right * Ax_left)' / z
    return gr, val
end

"""
    grad_squareloss_two_site(ψ::TensorTrain, k::Integer, X, Y) -> grad_sl, sl

Compute the gradient of the square loss from fitting data `(X,Y)` with the tensor train `ψ` with respect to the merged tensors Aᵏ and Aᵏ⁺¹.
Return also the loss, which is a byproduct of the computation.
"""
function grad_squareloss_two_site(ψ::TensorTrain, k::Integer, X, Y;
    prodA_left = [precompute_left_environments(ψ, x) for x in X],
    prodA_right = [precompute_right_environments(ψ, x) for x in X],
    Aᵏᵏ⁺¹ =_merge_tensors(ψ[k], ψ[k+1]),
    weight_decay = 0.0)

    T = length(X)
    @assert length(Y) == T
    @assert weight_decay >= 0
    # gA = zero(Aᵏᵏ⁺¹)
    gA = weight_decay * Aᵏᵏ⁺¹
    sl = weight_decay * abs2(norm(Aᵏᵏ⁺¹))

    # TODO: this operation is in principle parallelizable
    for (n,(x,y)) in enumerate(zip(X,Y))
        gr, val = grad_evaluate_two_site(ψ, k, x;
            Ax_left = prodA_left[n][k-1], Ax_right = prodA_right[n][k+2], Aᵏᵏ⁺¹
            )
        gA[:,:,x[k]...,x[k+1]...] .+= 1/T * gr * (val - y)
        sl += 1/T * abs2(val - y)
    end
    return gA, sl
end
