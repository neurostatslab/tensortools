"""
Diagnostic measures for CP decomposition fits.
"""

import numpy as np
from munkres import Munkres


def kruskal_align(U, V, permute_U=False, permute_V=False):
    """Aligns two KTensors and returns a similarity score.

    Parameters
    ----------
    U : KTensor
        First kruskal tensor to align.
    V : KTensor
        Second kruskal tensor to align.
    permute_U : bool
        If True, modifies 'U' to align the KTensors (default is False).
    permute_V : bool
        If True, modifies 'V' to align the KTensors (default is False).

    Notes
    -----
    If both `permute_U` and `permute_V` are both set to True, then the
    factors are ordered from most to least similar. If only one is
    True then the factors on the modified KTensor are re-ordered to
    match the factors in the un-aligned KTensor.

    Returns
    -------
    similarity : float
        Similarity score between zero and one.
    """

    # Compute similarity matrices.
    unrm = [f / np.linalg.norm(f, axis=0) for f in U.factors]
    vnrm = [f / np.linalg.norm(f, axis=0) for f in V.factors]
    sim_matrices = [np.dot(u.T, v) for u, v in zip(unrm, vnrm)]
    cost = 1 - np.mean(np.abs(sim_matrices), axis=0)

    # Solve matching problem via Hungarian algorithm.
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # Compute mean factor similarity given the optimal matching.
    similarity = np.mean(1 - cost[prmU, prmV])

    # If U and V are of different ranks, identify unmatched factors.
    unmatched_U = list(set(range(U.rank)) - set(prmU))
    unmatched_V = list(set(range(V.rank)) - set(prmV))

    # If permuting both U and V, order factors from most to least similar.
    if permute_U and permute_V:
        idx = np.argsort(cost[prmU, prmV])

    # If permute_U is False, then order the factors such that the ordering
    # for U is unchanged.
    elif permute_V:
        idx = np.argsort(prmU)

    # If permute_V is False, then order the factors such that the ordering
    # for V is unchanged.
    elif permute_U:
        idx = np.argsort(prmV)

    # If permute_U and permute_V are both False, then we are done and can
    # simply return the similarity.
    else:
        return similarity

    # Re-order the factor permutations.
    prmU = [prmU[i] for i in idx]
    prmV = [prmV[i] for i in idx]

    # Permute the factors.
    if permute_U:
        U.permute(prmU)
    if permute_V:
        V.permute(prmV)

    # Flip the signs of factors.
    flips = np.sign([F[prmU, prmV] for F in sim_matrices])
    flips[0] *= np.prod(flips, axis=0)  # always flip an even number of factors

    if permute_U:
        for i, f in enumerate(flips):
            U.factors[i] *= f

    elif permute_V:
        for i, f in enumerate(flips):
            V.factors[i] *= f

    # Return the similarity score
    return similarity
