"""
    PeriodicTensorTrain{F<:Number, N} <: AbstractTensorTrain{F,N}

A type for representing a Tensor Train with periodic boundary conditions
- `F` is the type of the matrix entries
- `N` is the number of indices of each tensor (2 virtual ones + `N-2` physical ones)
"""
struct PeriodicTensorTrain{F<:Number, N} <: AbstractPeriodicTensorTrain{F,N}
    tensors::Vector{Array{F,N}}

    function PeriodicTensorTrain{F,N}(tensors::Vector{Array{F,N}}) where {F<:Number, N}
        N > 2 || throw(ArgumentError("Tensors shold have at least 3 indices: 2 virtual and 1 physical"))
        size(tensors[1],1) == size(tensors[end],2) ||
            throw(ArgumentError("Number of rows of the first matrix should coincide with the number of columns of the last matrix"))
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

bond_dims(A::AbstractPeriodicTensorTrain) = [size(A[t], 1) for t in 1:lastindex(A)]

evaluate(A::AbstractPeriodicTensorTrain, X...) = tr(prod(@view a[:, :, x...] for (a,x) in zip(A, X...)))

function accumulate_L(A::AbstractPeriodicTensorTrain)
    l = [zeros(0,0) for _ in eachindex(A)]
    A⁰ = _reshape1(first(A))
    @reduce l⁰[a¹,a²] := sum(x) A⁰[a¹,a²,x]
    l[1] = l⁰

    lᵗ = l⁰
    for t in 1:length(A)-1
        Aᵗ = _reshape1(A[t+1])
        @reduce lᵗ[a¹,aᵗ⁺¹] |= sum(x,aᵗ) lᵗ[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] 
        l[t+1] = lᵗ
    end
    return l
end

function accumulate_R(A::AbstractPeriodicTensorTrain)
    r = [zeros(0,0) for _ in eachindex(A)]
    A⁰ = _reshape1(last(A))
    @reduce rᴸ[aᴸ,a¹] := sum(x) A⁰[aᴸ,a¹,x]
    r[end] = rᴸ

    rᵗ = rᴸ
    for t in length(A)-1:-1:1
        Aᵗ = _reshape1(A[t])
        @reduce rᵗ[aᵗ,a¹] |= sum(x,aᵗ⁺¹) Aᵗ[aᵗ,aᵗ⁺¹,x] * rᵗ[aᵗ⁺¹,a¹] 
        r[t] = rᵗ
    end
    return r
end

function marginals(A::PeriodicTensorTrain{F,N};
        l = accumulate_L(A), r = accumulate_R(A)) where {F<:Real,N}

    A¹ = _reshape1(A[begin]); r² = r[2]
    @reduce p¹[x] := sum(a¹,a²) A¹[a¹,a²,x] * r²[a²,a¹]
    p¹ ./= sum(p¹)
    p¹ = reshape(p¹, size(A[begin])[3:end])

    Aᴸ = _reshape1(A[end]); lᴸ⁻¹ = l[end-1]
    @reduce pᴸ[x] := sum(aᴸ,a¹) lᴸ⁻¹[a¹,aᴸ] * Aᴸ[aᴸ,a¹,x]
    pᴸ ./= sum(pᴸ)
    pᴸ = reshape(pᴸ, size(A[end])[3:end])

    p = map(2:length(A)-1) do t 
        lᵗ⁻¹ = l[t-1]
        Aᵗ = _reshape1(A[t])
        rᵗ⁺¹ = r[t+1]
        @reduce pᵗ[x] := sum(a¹,aᵗ,aᵗ⁺¹) lᵗ⁻¹[a¹,aᵗ] * Aᵗ[aᵗ,aᵗ⁺¹,x] * rᵗ⁺¹[aᵗ⁺¹,a¹]  
        pᵗ ./= sum(pᵗ)
        reshape(pᵗ, size(A[t])[3:end])
    end

    return append!([p¹], p, [pᴸ])
end

function twovar_marginals(A::AbstractPeriodicTensorTrain{F,N};
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

function normalization(A::PeriodicTensorTrain; l = accumulate_L(A), r = accumulate_R(A))
    z = tr(l[end])
    @assert tr(r[begin]) ≈ z "z=$z, got $(tr(r[begin])), A=$A"  # sanity check
    z
end

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

function StatsBase.sample!(rng::AbstractRNG, x, A::AbstractPeriodicTensorTrain{F,N};
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

function orthogonalize_right!(C::PeriodicTensorTrain; svd_trunc=TruncThresh(1e-6))
    C⁰ = _reshape1(C[begin])
    q = size(C⁰, 3)
    @cast M[m, (n, x)] := C⁰[m, n, x]
    U, λ, V = svd_trunc(M)
    @cast A⁰[m, n, x] := V'[m, (n, x)] x in 1:q
    C[begin] = _reshapeas(A⁰, C[begin])     
    Cᵗ⁻¹ = _reshape1(C[end])
    @tullio D[m, n, x] := Cᵗ⁻¹[m, k, x] * U[k, n] * λ[n]
    @cast M[m, (n, x)] := D[m, n, x]

    for t in length(C):-1:2
        U, λ, V = svd_trunc(M)
        @cast Aᵗ[m, n, x] := V'[m, (n, x)] x in 1:q
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
        @cast Aᵗ[m, n, x] := U[(m, x), n] x in 1:q
        A[t] = _reshapeas(Aᵗ, A[t])
        Aᵗ⁺¹ = _reshape1(A[t+1])
        @tullio D[m, n, x] := λ[m] * V'[m, l] * Aᵗ⁺¹[l, n, x]
        @cast M[(m, x), n] |= D[m, n, x]
    end
    U, λ, V = svd_trunc(M)
    @cast Aᵀ[m, n, x] := U[(m, x), n] x in 1:q
    A[end] = _reshapeas(Aᵀ, A[end])
    A⁰ = _reshape1(A[begin])
    @tullio D[m, n, x] := λ[m] * V'[m, l] * A⁰[l, n, x]
    A[begin] = _reshapeas(D,  A[begin])

    return A
end