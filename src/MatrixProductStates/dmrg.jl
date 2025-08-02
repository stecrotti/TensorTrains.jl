"""
Fit a MPS to data using the 2site-DMRG-like gradient descent.
The algorithm performs successive sweeps left->right, then right->left on the MPS.
At each step, at site k:
- The MPS is in canonical form (matrices 1:k-1 left-orthogonal, k+2:L right-orthogonal)
- Two adjacent matrices Aᵏ,Aᵏ⁺¹ are merged by matrix multiplication (that's why "2site"_DMRG)
- The gradient of the log-likelihood wrt the merged tensor is computed
- Some steps of gradient descent are performed
- The updated tensor is split back into two separate ones by SVD+truncations
- Matrix k (resp. k+1) is kept left(resp. right)-orthogonal when the sweep is going right (resp. left)
- Move to next site

Reference: https://arxiv.org/abs/1709.01662.
"""
function two_site_dmrg!(p, X, nsweeps; kw...)
    # Bring the MPS in canonical form wrt indices 1,2 to initiate the process
    orthogonalize_two_site_center!(p, 1; svd_trunc=TruncThresh(0.0))
    for sweep in 1:nsweeps
        # sweep left to right
        two_site_dmrg_sweep!(p, X, 1:length(p)-1, Left(); kw...)
        # sweep right to left
        two_site_dmrg_sweep!(p, X, length(p)-1:-1:1, Right(); kw...)
    end
end

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
        # The accumulation of matrices to the left and to the right 
        #  in `grad_loglikelihood_two_site` -> `grad_evaluate_two_site` takes O(L).
        # This should be avoidable by memorizing the product of matrices 1:k-1 and k+2:L 
        #  for all datapoints and at each step in the sweep updating the current (k-th) matrix.
        # Also, the checks on orthogonal form (is_canonical) are O(L) and should eventually 
        #  be dropped once we are sure that they always pass.
        # Everything within this loop should eventually be O(1) wrt L and O(nsamples)

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

