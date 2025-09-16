"""
    two_site_dmrg!(p::TensorTrain, X, Y, nsweeps; kw...)

Perform regression on data `(X, Y)` using a tensor train ansatz and the 2site-DMRG-like gradient descent.
"""
function two_site_dmrg!(p::TensorTrain, X, Y, nsweeps; kw...)
    func = (p, k, data; _kw...) -> grad_squareloss_two_site(p, k, data...; _kw...)
    return _two_site_dmrg_generic!(func, p, (X, Y), nsweeps; kw...)
end

"""
    _two_site_dmrg_generic!(func, p, data::Tuple, nsweeps; kw...)

Generic code to perform DMRG-like optimization sweeps on a Tensor Train-like object.
The specific function to be *minimized* is passed as the first argument.
`data` can either be:
- `(X,)`: for unsupervised learning (fitting a distribution)
- `(X, Y)`: for supervised learning (regression)

Note that this function is internal and should not be called directly. It is documented only for the purpose of understandability of the package's inner workings.

## Details

The algorithm performs successive sweeps left->right, then right->left on the MPS.
At each step, at site k:
- The TensorTrain/MPS is in canonical form (matrices 1:k-1 left-orthogonal, k+2:L right-orthogonal)
- Two adjacent matrices Aᵏ,Aᵏ⁺¹ are merged by matrix multiplication (that's why "2site"-DMRG)
- The gradient of the loss function wrt the merged tensor is computed
- Some steps of gradient descent are performed (by default using Adam)
- The updated tensor is split back into two separate ones by SVD+truncations
- Matrix k (resp. k+1) is kept left(resp. right)-orthogonal when the sweep is going right (resp. left)
- Move to next site

Adapted from https://arxiv.org/abs/1709.01662.
"""
function _two_site_dmrg_generic!(func, p, data::Tuple, nsweeps; kw...)
    X = data[1]
    for x in X
        is_in_domain(p, x...) ||
            throw(DomainError("The value x=$x in `X` exceed the domain of the Tensor Train"))
    end
    # Bring the MPS in canonical form wrt indices 1,2 to initiate the process
    orthogonalize_two_site_center!(p, 1; svd_trunc=TruncThresh(0.0))
    # Pre-compute left and right environments for efficient gradient calculations
    prodA_left = [precompute_left_environments(p, x) for x in X]
    prodA_right = [precompute_right_environments(p, x) for x in X]

    for sweep in 1:nsweeps
        # sweep left to right
        _two_site_dmrg_sweep_generic!(func, p, data, 1:length(p)-1, Left(), sweep;
            prodA_left, prodA_right, kw...)
        # sweep right to left
        _two_site_dmrg_sweep_generic!(func, p, data, length(p)-1:-1:1, Right(), sweep;
            prodA_left, prodA_right, kw...)
    end

end

function _two_site_dmrg_sweep_generic!(
    func,   # the function to be optimized f(p, k, data; kw...) -> grad, val
    p,
    data,    # data as tuple, either (X,) or (X,y)
    idxs,    # indices for the sweep
    lr::LeftOrRight,     # `Left()` if the sweep is going L->R, hence leaving behind left-orthogonal tensors. `Right()` otherwise
    sweep;  # the sweep number, used to print progress info
    svd_trunc = TruncBond(5),
    η = 1e-3,   # learning rate for gradient descent
    ndesc = 100,    # number of gradient descent steps
    optimizer = Optim.Adam(; alpha=η),
    callback = (sweep, k, p, loss_val) -> nothing,
    prodA_left = [precompute_left_environments(p, x) for x in data[1]],
    prodA_right = [precompute_right_environments(p, x) for x in data[1]],
    kw...)

    X = data[1]

    for k in idxs
        # Merge Aᵏ and Aᵏ⁺¹ into a larger tensor
        Aᵏᵏ⁺¹ = _merge_tensors(p[k], p[k+1])

        # make a function to be used by Optim.jl which computes value and gradient https://julianlsolvers.github.io/Optim.jl/stable/user/tipsandtricks/#Avoid-repeating-computations
        function make_fg(p, k, Aᵏᵏ⁺¹, data)
            function fg!(F, G, A)
                grad, val = func(p, k, data;
                    prodA_left, prodA_right, Aᵏᵏ⁺¹ = reshape(A, size(Aᵏᵏ⁺¹)),
                    kw...
                )
                if G !== nothing
                    G .= reshape(grad, length(G))
                end
                if F !== nothing
                    return val
                end
                return nothing
            end
        end

        res = Optim.optimize(
            Optim.only_fg!(make_fg(p, k, Aᵏᵏ⁺¹, data)),
            reshape(Aᵏᵏ⁺¹, length(Aᵏᵏ⁺¹)),
            optimizer,
            Optim.Options(iterations=ndesc)
        )

        Aᵏᵏ⁺¹ .= reshape(res.minimizer, size(Aᵏᵏ⁺¹))
        loss_val = res.minimum

        # Split the updated tensor into two by SVD + truncations
        p[k], p[k+1] = TensorTrains._split_tensor(Aᵏᵏ⁺¹; svd_trunc, lr)

        # Any post-update operations
        callback(sweep, k, p, loss_val)

        # Update environments for efficient computation of grad log-likelihood
        update_environments!(prodA_left, prodA_right, p, k, X, lr)
    end
end

"
Right after having updated Aᵏ by gradient descent in a left->right sweep,
update the memorized products of matrices 1 to k, store it in `prodA_left[k]`,
for all datapoints
"
function update_environments!(prodA_left, prodA_right, p, k::Integer, X, lr::Left)
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
function update_environments!(prodA_left, prodA_right, p, k::Integer, X, lr::Right)
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
