# Fit a MPS to data using the DMRG-like gradient descent as in https://arxiv.org/abs/1709.01662
function two_site_dmrg_sweep!(
    p::MPS,
    X,       # data as a vector of vectors
    idxs;    # indices for the sweep
    svd_trunc=TruncThresh(1e-6),    
    η = 1e-3,   # learning rate for gradient descent
    ndesc = 100,    # number of gradient descent steps
    callback = (it, p, k, ll) -> nothing)

    for k in idxs
        orthogonalize_two_site_center!(p, k; svd_trunc=TruncThresh(0.0))
        for it in 1:ndesc
            A = _merge_tensors(p[k], p[k+1])
            dlldA, ll = grad_loglikelihood_two_site(p, k, X)
            p[k], p[k+1] = TensorTrains._split_tensor(A + η*dlldA; svd_trunc)
            callback(it, p, k, ll)
        end
    end
end

function two_site_dmrg!(p, X, nsweeps; kw...)
    for sweep in 1:nsweeps
        two_site_dmrg_sweep!(p, X, 1:length(p)-1; kw...)
        two_site_dmrg_sweep!(p, X, length(p)-2:-1:1; kw...)
    end
end