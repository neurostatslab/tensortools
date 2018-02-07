"""
Core operations to align and score Kruskal tensors.
"""

import numpy as np
import itertools as itr
import ipdb

def normalize_factors(X):
    """
    Normalizes all factors to unit length, and returns
    factor weights (higher-order singular values.)

    Example:
        >>> Y, lam = normalize_factors(X)
        >>> [y * np.power(lam[None,:], 1/len(X)) for y in Y] # equal to X

    Parameters
    -----------
    X : list of ndarray
        list of factor matrices (each matrix has R
        columns, corresponding to the rank of the model)

    Returns
    -------
    Y : list of ndarray
        same as factors, but with normalized columns (unit length)
    lam : ndarray
        vector of length R holding the weight for each factor
    """
    factors, ndim, rank = _validate_factors(X)

    # factor norms
    lam = np.ones(rank)

    # destination for new ktensor
    newfactors = []

    # normalize columns of factor matrices
    lam = np.ones(rank)
    for fact in factors:
        s = np.linalg.norm(fact, axis=0)
        lam *= s
        newfactors.append(fact/(s+1e-20))

    return newfactors, lam

def standardize_factors(X, lam_ratios=None, sort_factors=True):
    """Sorts factors by norm and distributes factor weights across all modes

    Parameters
    ----------
    X : list of ndarray
        List of factor matrices (each matrix has R
        columns, corresponding to the rank of the model)
    lam_ratios (optinoal) : ndarray
        If specified, determines how to distribute factors weights. For example,
        if lam_ratios = [1, 0, 0, ...] then all factors are unit length except
        the first factor which is multiplied by the weight of that component.
    sort_factors (optional) : bool
        If True, sort the factors by their weight (significance).

    Returns
    -------
    Y : ndarray list
        list of factor matrices after standardization
    """

    # normalize tensor
    nrmfactors, lam = normalize_factors(X)

    # default to equally sized factors
    if lam_ratios is None:
        lam_ratios = np.ones(len(X))
    
    # check input is valid
    if len(lam_ratios) != len(X):
        raise ValueError('lam_ratios must be a list equal to the number of tensor modes/dimensions')
    elif np.min(lam_ratios) < 0:
        raise ValueError('lam_ratios must all be nonnegative')
    else:
        lam_ratios = np.array(lam_ratios) / np.sum(lam_ratios)

    # sort factors by norm
    if sort_factors:
        prm = np.argsort(lam)[::-1]
        return [f[:,prm]*np.power(lam[prm], r) for f, r in zip(nrmfactors, lam_ratios)]
    else:
        return [f*np.power(lam, r) for f, r in zip(nrmfactors, lam_ratios)]


def align_factors(A, B, penalize_lam=False):
    """Align two kruskal tensors.

    aligned_A, aligned_B, score = align_factors(A, B, **kwargs)

    Arguments
    ---------
    A : kruskal tensor
    B : kruskal tensor
    penalize_lam : bool (default=True)
        whether or not to penalize factor magnitudes

    Returns
    -------
    aligned_A : kruskal tensor
        aligned version of A
    aligned_B : kruskal tensor
        aligned version of B
    score : float
        similarity score between zero and one
    """

    # check tensor order matches
    ndim = len(A)
    if len(B) != ndim:
        raise ValueError('number of dimensions do not match.')

    # check tensor shapes match
    for a, b in zip(A, B):
        if a.shape[0] != b.shape[0]:
            raise ValueError('kruskal tensors do not have same shape.')

    # rank of A and B
    A, ndim_A, rank_A = _validate_factors(A)
    B, ndim_B, rank_B = _validate_factors(B)

    # check that factors aren't empty
    if (rank_A == 0) or (rank_B == 0):
        raise ValueError('Cannot align a tensor of rank zero')

    # function assumes rank(A) >= rank(B). Rather than raise an error, we make a recursive call.
    if rank_A < rank_B:
        aligned_B, aligned_A, score = align_factors(B, A, penalize_lam=penalize_lam)
        return aligned_A, aligned_B, score

    # compute inner product similarity matrix between factors
    A, lam_A = normalize_factors(A)
    B, lam_B = normalize_factors(B)
    dprod = np.array([np.dot(a.T, b) for a, b in zip(A, B)])
    sim = np.multiply.reduce([np.abs(dp) for dp in dprod])

    # include penalty on factor lengths
    if penalize_lam:
        for i, j in itr.product(range(rank_A), range(rank_B)):
            la, lb = lam_A[i], lam_B[j]
            sim[i, j] *= 1 - (abs(la-lb) / max(abs(la),abs(lb)))

    # find permutation of factors by a greedy method
    best_perm = rank_A*[None]
    for i, j in zip(*np.unravel_index(np.argsort(sim.ravel())[::-1], sim.shape)):
        if j not in best_perm:
            best_perm[i] = j
        elif best_perm[i] is None:
            best_perm[i] = j
        if None not in best_perm:
            break
    best_perm = np.array(best_perm)

    # total similarity score. If rank_A == rank_B, this is just the mean
    # similarity of the factors. If rank_A > rank_B, this is the mean
    # similarity across the factors in B matched to A (i.e. extra factors in A
    # are ignored).
    kth = rank_A - rank_B
    score = np.mean(np.partition(sim[np.arange(rank_A), best_perm], kth)[kth:])

    # Flip signs of ktensor factors for better alignment
    sgn = np.tile(np.power(lam_A, 1/ndim), (ndim,1))
    for j in range(rank_B):

        # factor i in A matched to factor j in B
        i = best_perm[j]

        # sort from least to most similar
        dpsrt = np.argsort(dprod[:, i, j])
        dp = dprod[dpsrt, i, j]

        # flip factors
        #   - need to flip in pairs of two
        #   - stop flipping once dp is positive
        for z in range(0, ndim-1, 2):
            if dp[z] >= 0 or abs(dp[z]) < dp[z+1]:
                break
            else:
                # flip signs
                sgn[dpsrt[z], i] *= -1
                sgn[dpsrt[z+1], i] *= -1

    # flip signs in A
    flipped_A = [s*a for s, a in zip(sgn, A)]
    aligned_B = [np.power(lam_B, 1/ndim)*b for b in B]

    # permute A to align with B
    assert np.all(best_perm >= 0)
    aligned_A = [a.copy()[:,best_perm] for a in flipped_A]
    return aligned_A, aligned_B, score

def _validate_factors(factors):
    """Checks that input is a valid kruskal tensor

    Returns
    -------
    ndim : int
        number of dimensions in tensor
    rank : int
        number of factors
    """
    ndim = len(factors)

    # if necessary, add an axis to factor matrices
    for i, f in enumerate(factors):
        if f.ndim == 1:
            factors[i] = f[:, np.newaxis]

    # check rank consistency
    rank = factors[0].shape[1]
    for f in factors:
        if f.shape[1] != rank:
            raise ValueError('KTensor has inconsistent rank along modes.')

    # return factors and info
    return factors, ndim, rank
