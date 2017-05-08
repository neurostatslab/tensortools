"""
Core operations to align and score Kruskal tensors.
"""

import numpy as np
import itertools as itr

def normalize_factors(factors):
    """Normalizes all factors to unit length
    """
    factors, ndim, rank = _validate_factors(factors)

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

def standardize_factors(factors, lam_ratios=None, sort_factors=True):
    """Sorts factors by norm

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        ie for all u in factor_matrices:
        u[i] has shape (s_u_i, R), where R is fixed
    mode: int
        mode of the desired unfolding

    Returns
    -------
    std_factors : ndarray list
        standardized Kruskal tensor with unit length factors
    """

    # normalize tensor
    nrmfactors, lam = normalize_factors(factors)

    # default to equally sized factors
    if lam_ratios is None:
        lam_ratios = np.ones(len(factors))
    
    # check input is valid
    if len(lam_ratios) != len(factors):
        raise ValueError('list of scalings must match the number of tensor modes/dimensions')
    elif np.min(lam_ratios) < 0:
        raise ValueError('list of scalings must be nonnegative')
    else:
        lam_ratios = np.array(lam_ratios) / np.sum(lam_ratios)

    # sort factors by norm
    if sort_factors:
        prm = np.argsort(lam)[::-1]
        return [f[:,prm]*np.power(lam[prm], r) for f, r in zip(nrmfactors, lam_ratios)]
    else:
        return [f*np.power(lam, r) for f, r in zip(nrmfactors, lam_ratios)]


def align_factors(A, B, greedy=None, penalize_lam=True):
    """Align two kruskal tensors.

    aligned_A, aligned_B, score = align_factors(A, B, **kwargs)

    Arguments
    ---------
    A : kruskal tensor
    B : kruskal tensor
    greedy : bool
        Whether to use a gredy algorithm to attempt alignment,
        or do an exhaustive search over all permutations.
        Defaults to True if rank >= 10, else defaults to False.
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

    # function assumes rank(A) >= rank(B). Rather than raise an error, we make a recursive call.
    if rank_A < rank_B:
        aligned_B, aligned_A, score = align_factors(B, A, greedy=greedy, penalize_lam=penalize_lam)
        return aligned_A, aligned_B, score

    # decide whether to use greedy method or exhaustive search
    if greedy is None:
        greedy = True if min(rank_A, rank_B) >= 10 else False

    A, lam_A = normalize_factors(A)
    B, lam_B = normalize_factors(B)

    # compute dot product
    dprod = np.array([np.dot(a.T, b) for a, b in zip(A, B)])

    # similarity matrix
    sim = np.multiply.reduce([np.abs(dp) for dp in dprod])

    # include penalty on factor lengths
    if penalize_lam:
        for i, j in itr.product(range(rank_A), range(rank_B)):
            la, lb = lam_A[i], lam_B[j]
            sim[i, j] *= 1 - (abs(la-lb) / max(abs(la),abs(lb)))

    if greedy:
        # find permutation of factors by a greedy method
        best_perm = -np.ones(rank_A, dtype='int')
        score = 0
        for r in range(rank_B):
            i, j = np.unravel_index(np.argmax(sim), sim.shape)
            score += sim[i,j]
            sim[i,:] = -1
            sim[:,j] = -1
            best_perm[j] = i
        score /= rank_B

    else:
        # search all permutations
        score = -1
        best_perm = np.arange(rank_A)
        for comb in itr.combinations(range(rank_A), rank_B):
            perm = -np.ones(rank_A, dtype='int')
            unset = list(set(range(rank_A)) - set(comb))
            perm[unset] = np.arange(rank_B, rank_A)
            for p in itr.permutations(comb):
                perm[list(comb)] = list(p)
                sc = sum([ sim[i,j] for j, i in enumerate(p)])
                if sc > score:
                    best_perm = perm.copy()
                    score = sc
        score /= rank_B

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
    aligned_B = [np.power(l, 1/ndim)*b for l, b in zip(lam_B, B)]

    # permute A to align with B
    aligned_A = [a[:,best_perm] for a in flipped_A]
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
