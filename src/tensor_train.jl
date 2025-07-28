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
function TensorTrain(tensors::Vector{<:AbstractArray{F,N}}; z=Logarithmic(one(F))) where {F<:Number, N} 
    return TensorTrain{F,N}(tensors; z)
end


@forward TensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, Base.lastindex,
    Base.setindex!, Base.length, Base.eachindex,
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
    rand_tt([T = Float64], bondsizes::AbstractVector{<:Integer}, q...)
    rand_tt([T = Float64], d::Integer, L::Integer, q...)

Construct a Tensor Train with `rand(T)` entries, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_tt(::Type{T}, bondsizes::AbstractVector{<:Integer}, q...) where T <: Number
    A = flat_tt(T, bondsizes, q...)
    foreach(a->(a .= rand.()), A)
    A
end

rand_tt(bondsizes::AbstractVector{<:Integer}, q...) = rand_tt(Float64, bondsizes, q...)

rand_tt(::Type{T}, d::Integer, L::Integer, q...) where {T <: Number} = rand_tt(T, [1; fill(d, L-1); 1], q...)

rand_tt(d::Integer, L::Integer, q...) = rand_tt(Float64, d, L, q...)


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


# used to do stuff like `A+B` with `A,B` tensor trains
function _compose(f, A::TensorTrain{F,NA}, B::TensorTrain{F,NB}) where {F,NA,NB}
    axes(A[1])[3:end] == axes(B[1])[3:end] || throw(ArgumentError("Tensor Trains must have the types of physical indices, got $(axes(A[1])[3:end]) and $(axes(B[1])[3:end])"))
    length(A) == length(B) || throw(ArgumentError("Tensor Trains must have the same length, got $(length(A)) and $(length(B))"))
    z = max(A.z, B.z)
    za, zb = float(A.z/z), f(float(B.z/z))
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

# TODO: return directly the grad of the log so the two z's will cancel out
# compute the gradient of evaluating the tensor train with respect to the entries of Aˡ
function grad_evaluate(A::TensorTrain, l::Integer, X)
    id = fill(one(eltype(A)), 1, 1)
    Ax_left = prod(@view a[:,:,x...] for (a,x) in zip(A[1:l-1], X[1:l-1]); init=id)
    Ax_right = reduce((A,B) -> B * A, @view a[:,:,x...] for (a,x) in zip(A[end:-1:l+1], X[end:-1:l+1]); init=id)
    Ax_center = A[l][:,:,X[l]...]
    z = float(A.z)
    val = only(Ax_left * Ax_center * Ax_right) / z
    gr = (Ax_right * Ax_left)' / z
    return gr, val
end