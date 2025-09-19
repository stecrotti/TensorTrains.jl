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

# d1 is small size, q is number of states
function Truncator(T,m,d,q)
    return Truncator(
        zeros(T,d,d,q), 
        (ones(T,m,d) for _ in 1:5)...,
        (ones(T,d,d) for _ in 1:5)...
    )
end

function truncate!(A1,A,t::Truncator; niter=10^4, niter2=1, tol=1e-12, damp=1e-1)
    P, Q, X, Y, S  = t.P, t.Q, t.X, t.Y, t.S 
    P1,Q1,X1,Y1,S1 = t.P1, t.Q1, t.X1, t.Y1, t.S1
    Anew = t.Anew

    P .= 1.0
    Q .= 1.0
    P1 .= 1.0
    Q1 .= 1.0

    function findeigen!(Z,S,X,AB; tol=1e-10)
        ε = 0.0
        for _ in 1:niter2
            S .= 0.0
            for (A,B) in AB
                mul!(X, Z, B)
                mul!(S, A, X, 1.0, 1.0)
            end
            normalize!(S)
            @tullio ε := (Z[i] - S[i])^2
            Z .= S
            ε < tol && break
        end
        ε
    end

    ε,εA = 0.0, 0.0
    for it in 1:niter
        ε=max(findeigen!(Q,X,S,@views ((A[:,:,x],A1[:,:,x]') for x in axes(A,3)); tol),
            findeigen!(P,X,S,@views ((A[:,:,x]',A1[:,:,x]) for x in axes(A,3)); tol),
            findeigen!(Q1,X1,S1,@views ((A1[:,:,x],A1[:,:,x]') for x in axes(A,3)); tol),
            findeigen!(P1,X1,S1,@views ((A1[:,:,x]',A1[:,:,x]) for x in axes(A,3)); tol))
        X1 .= P1; Y1 .= Q1
        X .= P; Y .= Q
        rdiv!(X, qr!(X1))
        rdiv!(Y, qr!(Y1))
        for x in axes(A,3)
            mul!(S, @view(A[:,:,x]), Y)
            mul!(@view(Anew[:,:,x]), X', S)
        end
        normalize!(Anew)
        εA = maximum(abs, (A1[i]-Anew[i] for i in eachindex(A1,Anew)))
        A1 .= damp .* A1 .+ (1-damp) .* Anew
        if max(ε,εA) < tol
            return A1,it,ε,εA
        end
    end
    A1,niter,ε,εA
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
    d       :: TI
    maxiter :: TI
    tol     :: TF
    damp    :: TF
end
TruncInfinite(d::Integer; maxiter=100, tol=1e-14, damp=1e-1) = TruncInfinite(d, maxiter, tol, damp)

summary(svd_trunc::TruncInfinite) = "VUMPS truncation to bond size m'="*string(svd_trunc.d)


function compress!(p::InfiniteUniformTensorTrain{F}; 
    svd_trunc::TruncInfinite=TruncInfinite(4), kw...) where F

    (; d, maxiter, tol, damp) = svd_trunc
    qs = size(p.tensor)[3:end]
    q = prod(qs)
    A = _reshape1(p.tensor)
    m = size(A, 1)
    @assert size(A, 2) == m
    tr = Truncator(F, m, d, q)
    init = randn(F, d, d, q)
    Anew_resh, niter, ε, εA = truncate!(init, A, tr; niter=maxiter, tol, damp)
    Anew = reshape(Anew_resh, d, d, qs...)
    p.tensor = Anew
end