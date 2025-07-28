# Gradient of Z w.r.t. Aˡ, and Z
function grad_normalization_canonical(p::MPS, l::Integer)
    @assert l <= length(p)
    @assert is_canonical(p, l)
    
    Aˡ = p[l]
    Aˡconj = conj(Aˡ)
    @tullio zz = Aˡconj[m,n,x1,x2] * Aˡ[m,n,x1,x2]
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    gradz = 2 * Aˡconj / z2
    return gradz, z
end

function loglikelihood(p::MPS, X)
    logz = log(normalization(p))
    return mean(log(evaluate(p, x)) for x in X) - logz 
end

function grad_loglikelihood(p::MPS, l::Integer, X)
    Zprime, Z = grad_normalization_canonical(p, l)
    ll = -log(Z)
    T = length(X)
    gA = - Zprime ./ Z
    for x in X 
        gr, val = grad_evaluate(p.ψ, l, x)
        gA[:,:,x[l]...] .+= 2/T * gr / val
        ll += 2/T * log(val)
    end
    return gA, ll
end

function merge_tensors(A, B)
    @assert size(A) == size(B)

end

# TODO: return directly the grad of the log so the two z's will cancel out
# Gradient of Z w.r.t. Aˡ and Aˡ⁺¹ or Aˡ⁻¹, and Z
function grad_normalization_canonical(p::MPS, l::Integer, lpm1::Integer)
    @assert abs(l - lpm1) == 1
    @assert l <= length(p)
    @assert is_canonical(p, l)
    
    Aˡ = p[l]
    Aˡconj = conj(Aˡ)
    @tullio zz = Aˡconj[m,n,x1,x2] * Aˡ[m,n,x1,x2]
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    gradz = 2 * Aˡconj / z2
    return gradz, z
end