# Fit a MPS to data using the DMRG-like gradient descent as in https://arxiv.org/abs/1709.01662
function two_site_dmrg_sweep!(
    p::MPS,
    X,       # data as a vector of vectors
    idxs,    # indices for the sweep
    lr::LeftOrRight;     # `Left()` if the sweep is going L->R, hence leaving behind left-orthogonal tensors. `Right()` otherwise  
    svd_trunc=TruncThresh(1e-6),    
    η = 1e-3,   # learning rate for gradient descent
    ndesc = 100,    # number of gradient descent steps
    callback = (it, p, k, ll) -> nothing)

    for k in idxs
        # TODO: performance improvements
        # The accumulation of matrices to the left and 
        #  to the right (hidden in `grad_loglikelihood_two_site` -> `grad_evaluate_two_site`)
        #  takes O(L). This should be avoidable by  memorizing stuff.
        # Also, the checks on orthogonal form are O(L) and should eventually be dropped.
        # Everything within this loop should eventually be O(1)

        for it in 1:ndesc
            A = _merge_tensors(p[k], p[k+1])
            # Compute the gradient wrt the merged pair of tensors
            dlldA, ll = grad_loglikelihood_two_site(p, k, X)
            # Do 1 step of gradient ascent
            Anew = A + η*dlldA
            # Split the updated tensor into two by SVD
            p[k], p[k+1] = TensorTrains._split_tensor(Anew; svd_trunc, lr)
            callback(it, p, k, ll)
        end
    end
end

function two_site_dmrg!(p, X, nsweeps; kw...)
    # Bring the MPS in canonical form wrt indices 1,2
    orthogonalize_two_site_center!(p, 1; svd_trunc=TruncThresh(0.0))
    for sweep in 1:nsweeps
        two_site_dmrg_sweep!(p, X, 1:length(p)-1, Left(); kw...)
        two_site_dmrg_sweep!(p, X, length(p)-1:-1:1, Right(); kw...)
    end
end