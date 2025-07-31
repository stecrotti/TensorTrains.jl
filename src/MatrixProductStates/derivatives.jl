# Gradient of Z w.r.t. Aᵏ, and Z
function grad_normalization_canonical(p::MPS, k::Integer)
    @assert k <= length(p)
    @assert is_canonical(p, k)
    
    Aᵏ = p[k]
    Aᵏconj = conj(Aᵏ)
    Aᵏ_ = _reshape1(Aᵏ)
    Aᵏconj_ = _reshape1(Aᵏconj) # group all the x's together
    @tullio zz = Aᵏconj_[m,n,x] * Aᵏ_[m,n,x]
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    gradz = conj(2 * Aᵏ / z2)
    return gradz, z
end

"""
    loglikelihood(p::MPS, X)

Compute the loglikelihood of the data `X` under the MPS distribution `p`.
"""
function loglikelihood(p::MPS, X)
    logz = log(normalization(p))
    return mean(log(evaluate(p, x)) for x in X) - logz 
end

# Gradient of loglikelihood w.r.t. Aᵏ, and loglikelihood
function grad_loglikelihood(p::MPS, k::Integer, X)
    Zprime, Z = grad_normalization_canonical(p, k)
    ll = -log(Z)
    T = length(X)
    gA = - Zprime ./ Z
    for x in X 
        gr, val = grad_evaluate(p.ψ, k, x)
        gA[:,:,x[k]...] .+= 2/T * gr / val
        ll += 2/T * log(val)
    end
    return gA, ll
end


# TODO: return directly the grad of the log so the two z's will cancel out

"""
    grad_normalization_two_site_canonical(p::MPS, k::Integer) -> gradz, z

Compute the gradient of the normalization of `p` with respect to the merged tensors Aᵏ and Aᵏ⁺¹.
Return also the normalization, which is a byproduct of the computation.
"""
function grad_normalization_two_site_canonical(p::MPS, k::Integer)
    @assert k <= length(p)
    @assert is_two_site_canonical(p, k)  # Ensure that the MPS is in canonical form wrt center sites k,k+1
    
    Aᵏ = p[k]
    Aᵏ⁺¹ = p[k+1]
    # Merge the two central tensors
    Aᵏᵏ⁺¹ = TensorTrains._merge_tensors(Aᵏ, Aᵏ⁺¹)
    # group the x's together
    Aᵏᵏ⁺¹_ = _reshape1(Aᵏᵏ⁺¹)
    # contract to compute (something proportional to) Z
    @tullio zz = conj(Aᵏᵏ⁺¹_[m,n,x]) * Aᵏᵏ⁺¹_[m,n,x]
    # recall that tensor trains have an overall multiplication factor stored in order to avoid numerical issues
    z2 = abs2(float(p.ψ.z))
    z = zz / z2
    gradz = conj(2 * Aᵏᵏ⁺¹ / z2)
    
    return gradz, z
end

"""
    grad_loglikelihood_two_site(p::MPS, k::Integer, X) -> grad_ll, ll

Compute the gradient of the loglikelihood of data `X` under the MPS distribution `p` with respect to the merged tensors Aᵏ and Aᵏ⁺¹.
Return also the loglikelihood, which is a byproduct of the computation. 
"""
function grad_loglikelihood_two_site(p::MPS, k::Integer, X)
    Zprime, Z = grad_normalization_two_site_canonical(p, k)
    ll = -log(Z) 
    T = length(X)
    gA = - Zprime ./ Z
    for x in X 
        gr, val = grad_evaluate_two_site(p.ψ, k, x)
        gA[:,:,x[k]...,x[k+1]...] .+= 2/T * gr / val
        ll += 2/T * log(val)
    end
    return gA, ll
end