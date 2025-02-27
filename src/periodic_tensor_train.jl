"""
    PeriodicTensorTrain{F<:Number, N} <: AbstractPeriodicTensorTrain{F,N}

A type for representing a Tensor Train with periodic boundary conditions
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
mutable struct PeriodicTensorTrain{F<:Number, N, T, Z} <: AbstractPeriodicTensorTrain{F,N}
    tensors::Vector{T}
    z::Z

    function PeriodicTensorTrain{F,N}(tensors::Vector{T}; z::Z=Logarithmic(one(F))) where {F<:Number, N, T<:AbstractArray{F,N},Z}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N,T,Z}(tensors, z)
    end

    PeriodicTensorTrain{F,N,T,Z}(tensors; z::Z=Logarithmic(one(F))) where {F,N,T,Z} = PeriodicTensorTrain{F,N}(tensors; z)
end
function PeriodicTensorTrain(tensors::Vector{T}; z=Logarithmic(one(F))) where {F<:Number, N, T<:AbstractArray{F,N}} 
    return PeriodicTensorTrain{F,N}(tensors; z)
end

@forward PeriodicTensorTrain.tensors Base.getindex, Base.iterate, Base.firstindex, 
    Base.lastindex, Base.setindex!, Base.length, Base.eachindex, 
    check_bond_dims

"""
    flat_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    flat_periodic_tt(d::Integer, L::Integer, q...)

Construct a (normalized) Tensor Train with periodic boundary conditions filled with a constant, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function flat_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    x = 1 / (prod(bondsizes)^(1/length(bondsizes))*prod(q))
    tensors = [fill(x, bondsizes[t], bondsizes[mod1(t+1,length(bondsizes))], q...) for t in eachindex(bondsizes)]
    PeriodicTensorTrain(tensors)
end
flat_periodic_tt(d::Integer, L::Integer, q...) = flat_periodic_tt(fill(d, L-1), q...)

"""
    rand_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    rand_periodic_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train with periodic boundary conditions with entries random in [0,1], by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function rand_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    PeriodicTensorTrain([rand(bondsizes[t], bondsizes[mod1(t+1,length(bondsizes))], q...) for t in eachindex(bondsizes)])
end
rand_periodic_tt(d::Integer, L::Integer, q...) = rand_periodic_tt(fill(d, L-1), q...)


function _compose(f, A::PeriodicTensorTrain{F,NA}, B::PeriodicTensorTrain{F,NB}) where {F,NA,NB}
    NA == NB || throw(ArgumentError("Tensor Trains must have the same number of variables, got $NA and $NB"))
    length(A) == length(B) || throw(ArgumentError("Tensor Trains must have the same length, got $(length(A)) and $(length(B))"))
    tensors = map(zip(eachindex(A),A,B)) do (t,Aᵗ,Bᵗ)
        sa = size(Aᵗ); sb = size(Bᵗ)
        if t == 1
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) f(Bᵗ[:,:,x...])] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        else
            Cᵗ = [ [Aᵗ[:,:,x...] zeros(sa[1],sb[2]); zeros(sb[1],sa[2]) Bᵗ[:,:,x...]] 
                for x in Iterators.product(axes(Aᵗ)[3:end]...)]
            reshape( reduce(hcat, Cᵗ), (sa .+ sb)[1:2]..., size(Aᵗ)[3:end]...)
        end
    end
    PeriodicTensorTrain(tensors)
end

PeriodicTensorTrain(A::TensorTrain) = PeriodicTensorTrain(A.tensors)


function orthogonalize_right!(C::PeriodicTensorTrain{F}; svd_trunc=TruncThresh(0.0)) where F
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[m, (n, x)] := C⁰[m, n, x]
    U, λ, V = svd_trunc(M)
    @cast A⁰[m, n, x] := V'[m, (n, x)] x ∈ 1:q
    C[begin] = _reshapeas(A⁰, C[begin])     
    Cᵗ⁻¹ = _reshape1(C[end])
    @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
    @cast M[m, (n, x)] := D[m, n, x]
    c = Logarithmic(one(F))

    for t in length(C):-1:2
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
    C[begin] = _reshapeas(D, C[begin])
    C.z /= c
    return C
end

function orthogonalize_left!(A::PeriodicTensorTrain{F}; svd_trunc=TruncThresh(0.0)) where F
    A⁰ = _reshape1(A[begin])
    q = size(A⁰, 3)
    @cast M[(m, x), n] |= A⁰[m, n, x]
    D = fill(1.0,1,1,1)  # initialize
    c = Logarithmic(one(F))

    for t in 1:length(A)-1
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x ∈ 1:q
        A[t] = _reshapeas(Aᵗ, A[t])
        Aᵗ⁺¹ = _reshape1(A[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Aᵗ⁺¹[l, n, x]
        m = maximum(abs, D)
        if !isnan(m) && !isinf(m) && !iszero(m)
            D ./= m
            c *= m
        end
        @cast M[(m, x), n] |= D[m, n, x]
    end
    U, λ, V = svd_trunc(M)
    @cast Aᵀ[m, n, x] := U[(m, x), n] x ∈ 1:q
    A[end] = _reshapeas(Aᵀ, A[end])
    A⁰ = _reshape1(A[begin])
    @tullio D[m, n, x] := λ[m] * V'[m, l] * A⁰[l, n, x]
    A[begin] = _reshapeas(D,  A[begin])
    A.z /= c
    return A
end
