_reshape1(x) = reshape(x, size(x,1), size(x,2), prod(size(x)[3:end])...)
_reshapeas(x,y) = reshape(x, size(x,1), size(x,2), size(y)[3:end]...)
_reshapeas(x,y::OffsetArray) = reshape(x, size(x,1), size(x,2), axes(y)[3:end]...)

# SAMPLING
# sample an index `i` of `w` with probability prop to `w[i]`
# copied from StatsBase but avoids creating a `Weight` object
# assumes the input vector is normalized
function sample_noalloc(rng::AbstractRNG, w) 
    t = rand(rng)
    i = 0
    cw = 0.0
    for p in w
        cw += p
        i += 1
        cw > t && return i
    end
    @assert false "$w"
end
sample_noalloc(w) = sample_noalloc(default_rng(), w)


function is_approx_identity(A; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps)
    idxs = Iterators.product([1:d for d in size(A)]...)
    for id in idxs
        if allequal(id) && !isapprox(A[id...],  1; atol, rtol)
            return false
        end
        if !allequal(id) && !isapprox(A[id...],  0; atol, rtol)
            return false
        end
    end
    return true
end