"""
    PeriodicTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train with periodic boundary conditions
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
struct PeriodicTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}
    tensors::Vector{Array{F,N}}

    function PeriodicTensorTrain{F,N}(tensors::Vector{Array{F,N}}) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        check_bond_dims(tensors) ||
            throw(ArgumentError("Matrix indices for matrix product non compatible"))
        return new{F,N}(tensors)
    end
end
function PeriodicTensorTrain(tensors::Vector{Array{F,N}}) where {F<:Number, N} 
    return PeriodicTensorTrain{F,N}(tensors)
end

@forward PeriodicTensorTrain.tensors getindex, iterate, firstindex, lastindex, setindex!, 
    check_bond_dims, length, eachindex

"""
    uniform_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    uniform_periodic_tt(d::Integer, L::Integer, q...)

Construct a Tensor Train with periodic boundary conditions full of 1's, by specifying either:
- `bondsizes`: the size of each bond
- `d` a fixed size for all bonds, `L` the length
and
- `q` a Tuple/Vector specifying the number of values taken by each variable on a single site
"""
function uniform_periodic_tt(bondsizes::AbstractVector{<:Integer}, q...)
    tensors = [ones(bondsizes[t], bondsizes[mod1(t+1,length(bondsizes))], q...) for t in eachindex(bondsizes)]
    PeriodicTensorTrain(tensors)
end
uniform_periodic_tt(d::Integer, L::Integer, q...) = uniform_periodic_tt(fill(d, L-1), q...)

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
    @assert NA == NB
    @assert length(A) == length(B)
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


function orthogonalize_right!(C::PeriodicTensorTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[m, (n, x)] := C⁰[m, n, x]
    U, λ, V = svd_trunc(M)
    @cast A⁰[m, n, x] := V'[m, (n, x)] x:q
    C[begin] = _reshapeas(A⁰, C[begin])     
    Cᵗ⁻¹ = _reshape1(C[end])
    @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
    @cast M[m, (n, x)] := D[m, n, x]

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x:q
        C[t] = _reshapeas(Aᵗ, C[t])     
        Cᵗ⁻¹ = _reshape1(C[t-1])
        @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
        @cast M[m, (n, x)] := D[m, n, x]
    end
    C[begin] = _reshapeas(D, C[begin])

    @assert check_bond_dims(C.tensors)

    return C
end

function orthogonalize_left!(A::PeriodicTensorTrain; svd_trunc=TruncThresh(1e-6))
    A⁰ = _reshape1(A[begin])
    q = size(A⁰, 3)
    @cast M[(m, x), n] |= A⁰[m, n, x]
    D = fill(1.0,1,1,1)  # initialize

    for t in 1:length(A)-1
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := U[(m, x), n] x:q
        A[t] = _reshapeas(Aᵗ, A[t])
        Aᵗ⁺¹ = _reshape1(A[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Aᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    U, λ, V = svd_trunc(M)
    @cast Aᵀ[m, n, x] := U[(m, x), n] x:q
    A[end] = _reshapeas(Aᵀ, A[end])
    A⁰ = _reshape1(A[begin])
    @tullio D[m, n, x] := λ[m] * V'[m, l] * A⁰[l, n, x]
    A[begin] = _reshapeas(D,  A[begin])

    return A
end