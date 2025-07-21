function exact_normalization(A::AbstractTensorTrain{F,N}) where {F,N}
    qs = [size(Aˡ)[3:end] for Aˡ in A]
    X = Iterators.product((1:prod(qˡ) for (Aˡ,qˡ) in zip(A,qs))...)
    return sum(
        evaluate(A, [Tuple(CartesianIndices(qs[l])[x[l]]) for l in eachindex(x)])
            for x in X; init=0.0
    )
end

function exact_prob(A::AbstractTensorTrain{F,N}) where {F,N}
    qs = [size(Aˡ)[3:end] for Aˡ in A]
    X = Iterators.product((1:prod(qˡ) for (Aˡ,qˡ) in zip(A,qs))...)
    map(X) do x
        evaluate(A, [Tuple(CartesianIndices(qs[l])[x[l]]) for l in eachindex(x)])
    end
end

function exact_marginals(A::AbstractTensorTrain{F,N}; 
        p = exact_prob(A), qs = [size(Aˡ)[3:end] for Aˡ in A]) where {F,N}
    map(eachindex(A)) do l
        v = sum(p, dims=eachindex(A)[Not(l)])
        mˡ = reshape(v, qs[l])
        mˡ ./= sum(mˡ)
        mˡ
    end
end

function exact_twovar_marginals(A::AbstractTensorTrain{F,N}; 
        p = exact_prob(A), qs = [size(Aˡ)[3:end] for Aˡ in A]) where {F,N}
    map(Iterators.product(eachindex(A), eachindex(A))) do (l,m)
        if l ≥ m
            zeros(zeros(Int, 2*(N-2))...)
        else
            v = sum(p, dims=eachindex(A)[Not(l,m)])
            pˡᵐ = reshape(v, qs[l]..., qs[m]...)
            pˡᵐ ./= sum(pˡᵐ)
            pˡᵐ
        end
    end
end

function exact_norm(A::AbstractTensorTrain{F,N}; p = exact_prob(A)) where {F,N}
    sqrt(sum(abs2, p))
end

function exact_dot(A::AbstractTensorTrain{F,N}, B::AbstractTensorTrain{F,N};
        pA = exact_prob(A), pB = exact_prob(B)) where {F,N}
    dot(pA, pB)
end

function exact_prob(p::TensorTrains.MatrixProductStates.MPS)
    qs = [size(Aˡ)[3:end] for Aˡ in p]
    X = Iterators.product((1:prod(qˡ) for (Aˡ,qˡ) in zip(p,qs))...)
    map(X) do x
        evaluate(p, [Tuple(CartesianIndices(qs[l])[x[l]]) for l in eachindex(x)])
    end
end