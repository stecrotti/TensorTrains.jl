_reshape1(x) = reshape(x, size(x,1), size(x,2), prod(size(x)[3:end])...)
_reshapeas(x,y) = reshape(x, size(x,1), size(x,2), size(y)[3:end]...)

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
    @assert false
end
sample_noalloc(w) = sample_noalloc(GLOBAL_RNG, w)