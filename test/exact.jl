function exact_prob(A::TensorTrain{F,N}) where {F,N}
    qs = [size(Aˡ)[3:end] for Aˡ in A]
    X = Iterators.product((1:prod(qˡ) for (Aˡ,qˡ) in zip(A,qs))...)
    map(X) do x
        evaluate(A, [Tuple(CartesianIndices(qs[l])[x[l]]) for l in eachindex(x)])
    end
end

function exact_marginals(A::TensorTrain{F,N}; 
        p = exact_prob(A), qs = [size(Aˡ)[3:end] for Aˡ in A]) where {F,N}
    map(eachindex(A)) do l
        v = sum(p, dims=eachindex(A)[Not(l)])
        mˡ = reshape(v, qs[l])
        mˡ ./= sum(mˡ)
        mˡ
    end
end