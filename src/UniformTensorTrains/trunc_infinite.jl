using LinearAlgebra, Tullio

function crude_trunc(A, d1)
    U,L,Vt = svd(reshape(A,size(A,1),size(A,2)*size(A,3)))
    R = @views reshape(Diagonal(L[1:d1])*(Vt')[1:d1,:], d1, size(A,1), size(A,3))
    U1 = @view U[:,1:d1]
    @tullio A1[i,j,x] := R[i,k,x]*U1[k,j]
end

struct Truncator{T}
    Anew::Array{T,3}
    P::Matrix{T}
    Q::Matrix{T}
    X::Matrix{T}
    Y::Matrix{T}
    S::Matrix{T}
    P1::Matrix{T}
    Q1::Matrix{T}
    X1::Matrix{T}
    Y1::Matrix{T}
    S1::Matrix{T}
end

Truncator(T,d,d1,k) = Truncator(
    zeros(T,d1,d1,k), 
    (ones(T,d,d1) for _ in 1:5)...,
    (ones(T,d1,d1) for _ in 1:5)...)

function apply!(S,AB,Z;X=similar(S))
    fill!(S, 0.0)
    for (A,B) in AB
        mul!(X, Z, B)
        mul!(S, A, X, 1.0, 1.0)
    end
    S
end

function findeigen!(Z, AB; S = similar(Z), X = similar(Z), tol2=1e-10, niter2=20)
    ε2 = zero(eltype(Z))
    for _ in 1:niter2
        apply!(S,AB,Z; X)
        normalize!(S), S, X
        ε2 = L2(Z,S)
        copyto!(Z, S)
        ε2 < tol2 && break
    end
    ε2
end

function L2(a, b)
    s = zero(eltype(a))
    @inbounds @simd for i in eachindex(a,b)
        s += abs2(a[i] - b[i])
    end
    return s
end


function truncate!(A1,A,t::Truncator; niter=10^4, niter2=1, tol=1e-12, tol2=1e-25, damp=0.99)
    P, Q, X, Y, S  = t.P, t.Q, t.X, t.Y, t.S 
    P1,Q1,X1,Y1,S1 = t.P1, t.Q1, t.X1, t.Y1, t.S1
    X1H, Y1H = Hermitian(X1), Hermitian(Y1)
    Anew = t.Anew

    fill!(P, 1.0); fill!(Q, 1.0); fill!(P1, 1.0); fill!(Q1, 1.0)

    @views AA1c,AcA1 = ((A[:,:,x],A1[:,:,x]') for x in axes(A,3)), ((A[:,:,x]',A1[:,:,x]) for x in axes(A,3))
    @views A1A1c,A1cA1 = ((A1[:,:,x],A1[:,:,x]') for x in axes(A,3)), ((A1[:,:,x]',A1[:,:,x]) for x in axes(A,3))

    ε::Float64 = 0.0
    εA::Float64 = 0.0
    for it in 1:niter
        ε = max(findeigen!(Q,AA1c; S, X, tol2, niter2), findeigen!(P,AcA1; S, X, tol2, niter2),
                findeigen!(Q1,A1A1c; S=S1, X=X1, tol2, niter2), findeigen!(P1,A1cA1; S=S1, X=X1, tol2, niter2))
        copyto!(X1, P1); copyto!(Y1, Q1); copyto!(X, P); copyto!(Y, Q)
        X1 .+= P1'; Y1 .+= Q1'
        X1 ./= 2.0; Y1 ./= 2.0
        if all(X1[i,i] > 0 for i in axes(X1,1)) && all(X1[i,i] > 0 for i in axes(X1,1))
            rdiv!(X, cholesky!(X1H, NoPivot())); rdiv!(Y, cholesky!(Y1H, NoPivot()))
        else
            rdiv!(X, lu!(X1)); rdiv!(Y, lu!(Y1))
        end
        for x in axes(A,3)
            apply!(@view(Anew[:,:,x]), ((X',Y),), @view(A[:,:,x]); X = S)
            #mul!(S, @view(A[:,:,x]), Y)
            #mul!(@view(Anew[:,:,x]), X', S) 
        end
        normalize!(Anew)
        εA = L2(A1,Anew)
        A1 .= damp .* A1 .+ (1.0 - damp) .* Anew
        max(ε,εA) < tol && return A1,it,ε,εA
    end
    A1,niter,ε,εA
end


function logdot(A,A1; niter2=10^4, tol=1e-100)
    @views AA1c = ((A[:,:,x],A1[:,:,x]') for x in axes(A,3))
    Q = fill(1.0, size(A,1), size(A1,1))
    findeigen!(Q, AA1c; tol2=tol, niter2)
    SQ = similar(Q)
    apply!(SQ, AA1c, Q)
    log(tr(SQ'Q))
end

function fidelity(A,A1; niter2 = 10^4, tol=1e-30)
    logdot(A,A1; niter2, tol)-logdot(A,A; niter2, tol)/2-logdot(A1,A1; niter2, tol)/2
end

"""
    TruncInfinite{TI, TF} <: SVDTrunc

A type used to perform  truncations of an [`InfiniteUniformTensorTrain`](@ref) to a target bond size `d`.

# FIELDS
- `d`: target bond dimension
- `maxiter = 100`: max number of iterations
- `tol = 1e-14`: tolerance 
- `damp = 1e-1`: damping

```@example
p = rand_infinite_uniform_tt(10, 2, 2)
compress!(p, TruncInfinite(5))
```
"""
struct TruncInfinite{TI<:Integer, TF<:Real} <: SVDTrunc
    d             :: TI
    maxiter_outer :: TI
    maxiter_inner :: TI
    tol_outer     :: TF
    tol_inner     :: TF
    damp          :: TF
end
function TruncInfinite(d::Integer; 
    maxiter_outer = 5*10^3, maxiter_inner = 40,
    tol_outer = 1e-20, tol_inner = 1e-50,
    damp=0.995)
    
    return TruncInfinite(d, maxiter_outer, maxiter_inner, tol_outer, tol_inner,
        damp)
end

summary(svd_trunc::TruncInfinite) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)


function compress!(p::InfiniteUniformTensorTrain{F}; 
    svd_trunc::TruncInfinite=TruncInfinite(4), kw...) where F

    (; d, maxiter_outer, maxiter_inner, tol_outer, tol_inner, damp) = svd_trunc
    qs = size(p.tensor)[3:end]
    q = prod(qs)
    A = _reshape1(p.tensor)
    m = size(A, 1)
    @assert size(A, 2) == m
    
    if m ≤ d
        return p
    end

    t = Truncator(F, m, d, q)
    init = crude_trunc(A, d)
    Anew_resh, _, _, _ = truncate!(init, A, t; 
        niter = maxiter_outer, niter2 = maxiter_inner,
        tol = tol_outer, tol2 = tol_inner, damp)
    Anew = reshape(Anew_resh, d, d, qs...)
    p.tensor = Anew
    return p
end