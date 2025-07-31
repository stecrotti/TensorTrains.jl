# Gradient of Z w.r.t. Aˡ, and Z
function grad_normalization_canonical(p::MPS, l::Integer)
    @assert l <= length(p)
    @assert is_canonical(p, l)
    
    Aˡ = p[l]
    Aˡconj = conj(Aˡ)
    Aˡ_ = _reshape1(Aˡ)
    Aˡconj_ = _reshape1(Aˡconj) # group all the x's together
    @tullio zz = Aˡconj_[m,n,x] * Aˡ_[m,n,x]
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    gradz = conj(2 * Aˡ / z2)
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


# TODO: return directly the grad of the log so the two z's will cancel out
# Gradient of Z w.r.t. Aˡ and Aˡ⁺¹ and Z
# For DMRG two-site update: the MPS should be in canonical form such that
# - A^1 to A^{l-1} are left-orthogonal
# - A^{l+2} to A^N are right-orthogonal  
# - A^l and A^{l+1} form the merged non-orthogonal center
function grad_normalization_two_site_canonical(p::MPS, l::Integer)
    @assert l <= length(p)
    @assert is_two_site_canonical(p, l)  # This ensures proper canonical form for two-site update
    
    Aˡ = p[l]
    Aˡ⁺¹ = p[l+1]
    
    # Compute the merged tensor and its normalization 
    Aˡˡ⁺¹ = TensorTrains._merge_tensors(Aˡ, Aˡ⁺¹)
    Aˡˡ⁺¹_ = _reshape1(Aˡˡ⁺¹)
    @tullio zz = conj(Aˡˡ⁺¹_[m,n,x]) * Aˡˡ⁺¹_[m,n,x] 
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    
    # The gradient with respect to the merged tensor is:
    gradz = conj(2 * Aˡˡ⁺¹ / z2)
    
    return gradz, z
end