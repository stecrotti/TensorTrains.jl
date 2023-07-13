var documenterSearchIndex = {"docs":
[{"location":"guide/#Types","page":"Guide","title":"Types","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"AbstractTensorTrain\nTensorTrain","category":"page"},{"location":"guide/#TensorTrains.AbstractTensorTrain","page":"Guide","title":"TensorTrains.AbstractTensorTrain","text":"AbstractTensorTrain\n\nAn abstract type representing a Tensor Train. Currently, there is only one concrete subtype TensorTrain.\n\n\n\n\n\n","category":"type"},{"location":"guide/#TensorTrains.TensorTrain","page":"Guide","title":"TensorTrains.TensorTrain","text":"TensorTrain{F<:Number, N} <: AbstractTensorTrain\n\nA type for representing a Tensor Train\n\nF is the type of the matrix entries\nN is the number of indices of each tensor (2 virtual ones + N-2 physical ones)\n\n\n\n\n\n","category":"type"},{"location":"guide/#Functions","page":"Guide","title":"Functions","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"normalize_eachmatrix!\nuniform_tt\nrand_tt\nbond_dims\nevaluate\nmarginals\ntwovar_marginals\nnormalization\nnormalize!(::TensorTrain)\n+\n-\nsample!\nsample\northogonalize_right!\northogonalize_left!\ncompress!","category":"page"},{"location":"guide/#TensorTrains.normalize_eachmatrix!","page":"Guide","title":"TensorTrains.normalize_eachmatrix!","text":"normalize_eachmatrix!(A::TensorTrain)\n\nDivide each matrix by its maximum (absolute) element and return the sum of the logs of the individual normalizations. This is used to keep the entries from exploding during computations\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.uniform_tt","page":"Guide","title":"TensorTrains.uniform_tt","text":"uniform_tt(bondsizes::AbstractVector{<:Integer}, q...)\nuniform_tt(d::Integer, L::Integer, q...)\n\nConstruct a Tensor Train full of 1's, by specifying either:\n\nbondsizes: the size of each bond\nd a fixed size for all bonds, L the length\n\nand\n\nq a Tuple/Vector specifying the number of values taken by each variable on a single site\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.rand_tt","page":"Guide","title":"TensorTrains.rand_tt","text":"rand_tt(bondsizes::AbstractVector{<:Integer}, q...)\nrand_tt(d::Integer, L::Integer, q...)\n\nConstruct a Tensor Train with entries random in [0,1], by specifying either:\n\nbondsizes: the size of each bond\nd a fixed size for all bonds, L the length\n\nand\n\nq a Tuple/Vector specifying the number of values taken by each variable on a single site\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.bond_dims","page":"Guide","title":"TensorTrains.bond_dims","text":"bond_dims(A::TensorTrain)\n\nReturn a vector with the dimensions of the virtual bonds\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.evaluate","page":"Guide","title":"TensorTrains.evaluate","text":"evaluate(A::TensorTrain, X...)\n\nEvaluate the Tensor Train A at input X\n\nExample:\n\n    L = 3\n    q = (2, 3)\n    A = rand_tt(4, L, q...)\n    X = [[rand(1:qi) for qi in q] for l in 1:L]\n    evaluate(A, X)\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.marginals","page":"Guide","title":"TensorTrains.marginals","text":"marginals(A::TensorTrain; l, r)\n\nCompute the marginal distributions p(x^l) at each site\n\nOptional arguments\n\nl = accumulate_L(A), r = accumulate_R(A) pre-computed partial nommalizations\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.twovar_marginals","page":"Guide","title":"TensorTrains.twovar_marginals","text":"marginals(A::TensorTrain; l, r, M, Δlmax)\n\nCompute the marginal distributions for each pair of sites p(x^l x^m)\n\nOptional arguments\n\nl = accumulate_L(A), r = accumulate_R(A), M = accumulate_M(A) pre-computed partial normalizations\nΔmax=length(A): compute marginals only at distance Δmax: l-mle Δmax\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.normalization","page":"Guide","title":"TensorTrains.normalization","text":"normalization(A::TensorTrain; l, r)\n\nCompute the normalization Z=sum_x^1ldotsx^L A^1(x^1)cdots A^L(x^L)\n\n\n\n\n\n","category":"function"},{"location":"guide/#LinearAlgebra.normalize!-Tuple{TensorTrain}","page":"Guide","title":"LinearAlgebra.normalize!","text":"normalize!(A::TensorTrain)\n\nNormalize A to a probability distribution\n\n\n\n\n\n","category":"method"},{"location":"guide/#Base.:+","page":"Guide","title":"Base.:+","text":"+(A::TensorTrain, B::TensorTrain)\n\nCompute the sum of two Tensor Trains. Matrix sizes are doubled\n\n\n\n\n\n","category":"function"},{"location":"guide/#Base.:-","page":"Guide","title":"Base.:-","text":"-(A::TensorTrain, B::TensorTrain)\n\nCompute the difference of two Tensor Trains. Matrix sizes are doubled\n\n\n\n\n\n","category":"function"},{"location":"guide/#StatsBase.sample!","page":"Guide","title":"StatsBase.sample!","text":"sample!([rng], x, A::TensorTrain; r)\n\nDraw an exact sample from A and store the result in x.\n\nOptionally specify a random number generator rng as the first argument   (defaults to Random.GLOBAL_RNG) and provide a pre-computed r = accumulate_R(A).\n\nThe output is x,p, the sampled sequence and its probability\n\n\n\n\n\n","category":"function"},{"location":"guide/#StatsBase.sample","page":"Guide","title":"StatsBase.sample","text":"sample([rng], A::TensorTrain; r)\n\nDraw an exact sample from A.\n\nOptionally specify a random number generator rng as the first argument   (defaults to Random.GLOBAL_RNG) and provide a pre-computed r = accumulate_R(A).\n\nThe output is x,p, the sampled sequence and its probability\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.orthogonalize_right!","page":"Guide","title":"TensorTrains.orthogonalize_right!","text":"orthogonalize_right!(A::TensorTrain; svd_trunc::SVDTrunc)\n\nBring A to right-orthogonal form by means of SVD decompositions.\n\nOptionally perform truncations by passing a SVDTrunc.\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.orthogonalize_left!","page":"Guide","title":"TensorTrains.orthogonalize_left!","text":"orthogonalize_left!(A::TensorTrain; svd_trunc::SVDTrunc)\n\nBring A to left-orthogonal form by means of SVD decompositions.\n\nOptionally perform truncations by passing a SVDTrunc.\n\n\n\n\n\n","category":"function"},{"location":"guide/#TensorTrains.compress!","page":"Guide","title":"TensorTrains.compress!","text":"compress!(A::TensorTrain; svd_trunc::SVDTrunc)\n\nCompress A by means of SVD decompositions + truncations\n\n\n\n\n\n","category":"function"},{"location":"guide/#SVD-Truncators","page":"Guide","title":"SVD Truncators","text":"","category":"section"},{"location":"guide/","page":"Guide","title":"Guide","text":"SVDTrunc\nTruncThresh\nTruncBond\nTruncBondMax\nTruncBondThresh","category":"page"},{"location":"guide/#TensorTrains.SVDTrunc","page":"Guide","title":"TensorTrains.SVDTrunc","text":"abstract type SVDTrunc\n\nSVD truncator. Can be threshold-based or bond size-based\n\n\n\n\n\n","category":"type"},{"location":"guide/#TensorTrains.TruncThresh","page":"Guide","title":"TensorTrains.TruncThresh","text":"TruncThresh{T} <: SVDTrunc\n\nA type used to perform SVD-based truncations based on a threshold ε. Given a vector lambda of m singular values, those below varepsilonsqrtsum_k=1^m lambda_k^2 are truncated to zero.\n\nFIELDS\n\nε: threshold.\n\nsvd_trunc = TruncThresh(1e-5)\nM = rand(5,6)\nM_trunc = svd_trunc(M)\n\n\n\n\n\n","category":"type"},{"location":"guide/#TensorTrains.TruncBond","page":"Guide","title":"TensorTrains.TruncBond","text":"TruncBond{T} <: SVDTrunc\n\nA type used to perform SVD-based truncations based on bond size m'. Given a vector lambda of m singular values, only the m largest are kept, the others are truncated to zero.\n\nFIELDS\n\nmprime: number of singular values to retain\n\nsvd_trunc = TruncBond(3)\nM = rand(5,6)\nM_trunc = svd_trunc(M)\n\n\n\n\n\n","category":"type"},{"location":"guide/#TensorTrains.TruncBondMax","page":"Guide","title":"TensorTrains.TruncBondMax","text":"TruncBondMax{T} <: SVDTrunc\n\nSimilar to TruncBond, but also stores the maximum error sqrtfracsum_k=m+1^mlambda_k^2sum_k=1^mlambda_k^2 made since the creation of the object\n\nFIELDS\n\nmprime: number of singular values to retain\nmaxerr: a 1-element vector storing the maximum error\n\n\n\n\n\n","category":"type"},{"location":"guide/#TensorTrains.TruncBondThresh","page":"Guide","title":"TensorTrains.TruncBondThresh","text":"TruncBondThresh{T} <: SVDTrunc\n\nA mixture of TruncBond and TruncThresh, truncates to the most stringent criterion.\n\n\n\n\n\n","category":"type"},{"location":"#What-is-a-Tensor-Train?","page":"Home","title":"What is a Tensor Train?","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A Tensor Train is a type of tensor factorization involving the product of 3-index tensors organized on a one-dimensional chain.  In the context of function approximation and probability, a function of L discrete variables is in Tensor Train format if it is written as","category":"page"},{"location":"","page":"Home","title":"Home","text":"f(x^1 x^2 ldots x^L) = sum_a^1a^2ldotsa^L-1 A^1(x^1)_a^1A^2(x^2)_a^1a^2cdots A^L-1(x^L-1)_a^L-2a^L-1A^L(x^L)_a^L-1","category":"page"},{"location":"","page":"Home","title":"Home","text":"where, for every choice of x^l, A^l(x^l) is a real-valued matrix and the matrix sizes must be compatible. The first matrix must have 1 row and the last matrix should have 1 column, such that the product correctly returns a scalar.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Tensor Train factorization can be used to parametrize probability distributions, which is the main focus of this package. In this case, f should be properly normalized and always return a non-negative value. ","category":"page"},{"location":"#Notation-and-terminology","page":"Home","title":"Notation and terminology","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tensor Trains are the most basic type of Tensor Network. Tensor networks are a large family of tensor factorizations which are often best represented in diagrammatic notation. For this reason, the term bond is used interchangeably as index. The indices a^1a^2ldotsa^L-1 are usually called the virtual indices, while x^1 x^2 ldots x^L are the physical indices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Tensor Trains are used to parametrize wavefunctions in many-body quantum physics. The resulting quantum state is called Matrix Product State. In such context, the entries are generally complex numbers, and a probability can be obtained for a given state by taking the squared absolute value of the wavefunction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In this package we focus on the \"classical\" case where the Tensor Train directly represents a probability distribution p(x^1 x^2 ldots x^L). ","category":"page"},{"location":"#Efficient-computation","page":"Home","title":"Efficient computation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Given a Tensor Train some simple recursive strategies can be employed to","category":"page"},{"location":"#Compute-the-normalization","page":"Home","title":"Compute the normalization","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Z = sum_x^1 x^2 ldots x^L sum_a^1a^2ldotsa^L-1 A^1(x^1)_a^1A^2(x^2)_a^1a^2cdots A^L-1(x^L-1)_a^L-2a^L-1A^L(x^L)_a^L-1","category":"page"},{"location":"","page":"Home","title":"Home","text":"such that ","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginaligned\n1=sum_x^1 x^2 ldots x^Lp(x^1 x^2 ldots x^L)=sum_x^1 x^2 ldots x^Lfrac1Z sum_a^1a^2ldotsa^L-1 A^1(x^1)_a^1A^2(x^2)_a^1a^2cdots A^L-1(x^L-1)_a^L-2a^L-1A^L(x^L)_a^L-1\nendaligned","category":"page"},{"location":"#Compute-marginals","page":"Home","title":"Compute marginals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Single-variable","category":"page"},{"location":"","page":"Home","title":"Home","text":"p(x^l=x) = sum_x^1 x^2 ldots x^L p(x^1 x^2 ldots x^L) delta(x^lx)","category":"page"},{"location":"","page":"Home","title":"Home","text":"and two-variable","category":"page"},{"location":"","page":"Home","title":"Home","text":"p(x^l=x x^m=x) = sum_x^1 x^2 ldots x^L p(x^1 x^2 ldots x^L) delta(x^lx)delta(x^mx)","category":"page"},{"location":"#Extract-exact-samples","page":"Home","title":"Extract exact samples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Via hierarchical sampling","category":"page"},{"location":"","page":"Home","title":"Home","text":"p(x^1 x^2 ldots x^L) = p(x^1)p(x^2x^1)p(x^3x^1x^2)cdots p(x^Lx^1x^2ldotsx^L-1)","category":"page"},{"location":"","page":"Home","title":"Home","text":"by first sampling x^1sim p(x^1), then x^2sim p(x^2x^1) and so on.","category":"page"},{"location":"#What-can-this-package-do?","page":"Home","title":"What can this package do?","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This small package provides some utilities for creating, manipulating and evaluating Tensor Trains interpreted as functions, with a focus on the probabilistic side.  Each variable x^l is assumed to be multivariate. Whenever performing some probability-related operation, it is responsability of the user to make sure that the Tensor Train always represents a valid probability distribution.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Common operations are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"evaluate a Tensor Train at a given set of indices\northogonalize_left!, orthogonalize_right!: bring a Tensor Train to left/right orthogonal form\ncompress! a Tensor Train using SVD-based truncations\nnormalize! a Tensor Train in the probability sense (not in the L_2 norm sense!), see above\nsample from a Tensor Train intended as a probability ditribution, see above\n+,-: take the sum/difference of two TensorTrains","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Let's construct and initialize at random a Tensor Train of the form","category":"page"},{"location":"","page":"Home","title":"Home","text":"fleft((x^1y^1) (x^2y^2) (x^3y^3)right) = sum_a^1a^2 A^1(x^1y^1)_a^1A^2(x^2y^2)_a^1a^2A^3(x^3y^3)_a^2","category":"page"},{"location":"","page":"Home","title":"Home","text":"where x^lin12 y^lin123.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using TensorTrains\nL = 3        # length\nq = (2, 3)   # number of values taken by x, y\nd = 5        # bond dimension\nA = rand_tt(d, L, q...)    # construct Tensor Train with random positive entries\nxy = [[rand(1:qi) for qi in q] for _ in 1:L]    # random set of indices\np = evaluate(A, xy)    # evaluate `A` at `xy`\ncompress!(A; svd_trunc=TruncThresh(1e-8));    # compress `A` to reduce the bond dimension\npnew = evaluate(A, xy)\nε = abs((p-pnew)/p)","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"https://tensornetwork.org: \"an open-source review article focused on tensor network algorithms, applications, and software\"\nOseledets, I.V., 2011. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5).","category":"page"}]
}
