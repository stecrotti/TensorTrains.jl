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
    all(is_in_domain(p, x...) for x in X) ||
        throw(DomainError("The values in `X` exceed the domain of the MPS"))
    # Bring the MPS in canonical form wrt indices 1,2 to initiate the process
    orthogonalize_two_site_center!(p, 1; svd_trunc=TruncThresh(0.0))
    # Pre-compute left and right environments for efficient gradient calculations
    prodA_left = [precompute_left_environments(p.ψ, x) for x in X]
    prodA_right = [precompute_right_environments(p.ψ, x) for x in X]

    for sweep in 1:nsweeps
        # sweep left to right
        two_site_dmrg_sweep!(p, X, 1:length(p)-1, Left(), sweep; 
            prodA_left, prodA_right, kw...)
        # sweep right to left
        two_site_dmrg_sweep!(p, X, length(p)-1:-1:1, Right(), sweep; 
            prodA_left, prodA_right, kw...)
    end

end

function two_site_dmrg_sweep!(
    p::MPS,
    X,       # data as a vector of vectors
    idxs,    # indices for the sweep
    lr::LeftOrRight,     # `Left()` if the sweep is going L->R, hence leaving behind left-orthogonal tensors. `Right()` otherwise 
    sweep;  # the sweep number, used to print progress info 
    svd_trunc = TruncBond(5),    
    η = 1e-3,   # learning rate for gradient descent
    ndesc = 100,    # number of gradient descent steps
    callback = (sweep, k, it, p, ll) -> nothing,
    prodA_left = [precompute_left_environments(p.ψ, x) for x in X],
    prodA_right = [precompute_right_environments(p.ψ, x) for x in X])


    for k in idxs
        # TODO: optimizers
        # Instead of writing gradient descent by hand, use some optimization
        #  library. This will allow to explore other optimizers

        # Merge Aᵏ and Aᵏ⁺¹ into a larger tensor
        A = _merge_tensors(p[k], p[k+1]) 

        for it in 1:ndesc
            # Compute the gradient wrt the merged tensors
            dlldA, ll = grad_loglikelihood_two_site(p, k, X;
                prodA_left, prodA_right
                )
            # Do 1 step of gradient ascent
            A .+= η*dlldA
            # Split the updated tensor into two by SVD + truncations
            p[k], p[k+1] = TensorTrains._split_tensor(A; svd_trunc, lr)
            # Any post-update operations
            callback(sweep, k, it, p, ll)
        end

        # Update environments for efficient computation of grad log-likelihood
        update_environments!(prodA_left, prodA_right, p, k, X, lr)
    end
end

"
Right after having updated Aᵏ by gradient descent in a left->right sweep,
update the memorized products of matrices 1 to k, store it in `prodA_left[k]`, 
for all datapoints
"
function update_environments!(prodA_left, prodA_right, p::MPS, k::Integer, X, lr::Left)
    for (n,x) in enumerate(X)
        Anew = p[k][:,:,x[k]...]
        prodA_left[n][k] = if k == 1
            Anew
        else
            prodA_left[n][k-1] * Anew
        end
    end
    return nothing
end

"
Right after having updated Aᵏ⁺¹ by gradient descent in a right->left sweep,
update the memorized products of matrices k+1 to L, store it in `prodA_right[k+1]`, 
for all datapoints
"
function update_environments!(prodA_left, prodA_right, p::MPS, k::Integer, X, lr::Right)
    for (n,x) in enumerate(X)
        Anew = p[k+1][:,:,x[k+1]...]
        prodA_right[n][k+1] = if k == length(p)-1
            Anew
        else
            Anew * prodA_right[n][k+2]
        end
    end
    return nothing
end