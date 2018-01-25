def admm_solver(X0, A, B, gam1, gam2, nonneg, lam=1.0, iterations=1000):
    """Minimizes ||X*A - B||_F + elastic_net(X) for CP decomposition subproblem

    Parameters
    ----------
    X0 : ndarray
            n x r matrix, initial guess for X
    A : ndarray
            r x r, symmetric matrix holding reduced Grammians
    B : ndarray
            n x r matrix, unfolding times khatri-rao product
    gam1 : float
            strength of L1 penalty on X
    gam2 : float
            strength of L2 penalty on X
    nonneg : bool
            If true, restrict X to have nonnegative entries
    lam : float
            ADMM parameter
    iterations : int
            maximum number of iterations before we quit
    """

    # admm penalty param
    lam1 = gam1*lam
    lam2 = gam2*lam

    # cache lu factorization for fast prox operator
    # add 1/lam to diagonal of AtA
    Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

    # proximal operators
    prox_f = lambda v: scipy.linalg.lu_solve(Afct, (B + v/lam).T).T
    if nonneg:
        prox_g = lambda v: np.maximum(0, v-lam1) / (1 + lam2)
    else:
        prox_g = lambda v: (np.maximum(0, v-lam1) - np.maximum(0, -v-lam1)) / (1 + lam2)

    # initialize admm
    x = X0.copy()
    z = prox_g(x)
    u = x - z

    # admm iterations
    for itr in range(iterations):
        # updates
        x1 = prox_f(z - u)
        z1 = prox_g(x1 + u)
        u1 = u + x1 - z1

        # primal resids (r) and dual resids (s)
        r = np.linalg.norm(x1 - z1)
        s = (1/lam) * np.linalg.norm(z - z1)

        # # keep primal and dual resids within factor of 10
        # if r > 10*s:
        #     lam = lam / 2
        #     # print('{} - {} - {}'.format(itr, r, s))
        #     lam1 = gam1*lam
        #     lam2 = gam2*lam
        #     Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

        # elif s > 10*r:
        #     lam = lam * 1.9
        #     # print('{} * {} * {}'.format(itr, r, s))
        #     lam1 = gam1*lam
        #     lam2 = gam2*lam
        #     Afct = scipy.linalg.lu_factor(_add_to_diag(A, 1/lam))

        # accept parameter update
        x, z, u = x1.copy(), z1.copy(), u1.copy()

        # quit if we've converged
        if r < np.sqrt(x.size)*1e-3 and s < np.sqrt(x.size)*1e-3:
            break

    return x

def _l1_reg(lam, X):
    """Returns value and gradient of l1 regularization term on X

    Parameters
    ----------
    lam : float
        scale of the regularization
    X : ndarray
        Array holding the optimized variables
    """
    if lam is None:
        return 0, 0
    else:
        f = lam * np.sum(np.abs(X))
        g = lam * np.sign(X)
        return f, g

def _l2_reg(lam, X):
    """Returns value and gradient of l2 regularization term on X

    Parameters
    ----------
    lam : float
        scale of the regularization
    X : ndarray
        Array holding the optimized variables
    """
    if lam is None:
        return 0, 0
    else:
        f = 0.5 * lam * np.sum(X**2)
        g = lam * X
        return f, g

def _add_to_diag(A, z):
    """Add z to diagonal of matrix A.
    """
    B = A.copy()
    B[np.diag_indices_from(B)] + z
    return B
