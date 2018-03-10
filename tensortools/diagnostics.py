"""
Diagnostic measures for CP decomposition fits
"""

import numpy as np
from copy import deepcopy
from munkres import Munkres


def kruskal_align(U, V, permute_U=False, permute_V=False):
    """Aligns

    Parameters
    ----------
    U : KTensor
        First kruskal tensor to align.
    V : KTensor
        Second kruskal tensor to align.
    inplace : bool
        If True, overwrite inputs otherwise create copies (default: False).

    Returns
    -------
    similarity : float
        Similarity score between zero and one.
    """

    # matching cost from U to V
    unrm = [f / np.linalg.norm(f, axis=0) for f in U.factors]
    vnrm = [f / np.linalg.norm(f, axis=0) for f in V.factors]
    sim_matrices = [np.dot(u.T, v) for u, v in zip(unrm, vnrm)]
    cost = 1 - np.multiply.reduce(np.abs(sim_matrices), axis=0)

    # solve matching problem via Hungarian algorithm
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # If U and V are of different ranks, add unmatched factors to end.
    if U.rank > V.rank:
        prmU = np.append(prmU, list(set(range(U.rank)) - set(prmU)))
    elif U.rank < V.rank:
        prmV = np.append(prmV, list(set(range(V.rank)) - set(prmV)))

    # compute similarity across all factors
    similarity = np.mean(1 - cost[prmU, prmV])

    # Permute U and V to order factors from most to least similar.
    if permute_U and permute_V:
        idx = np.argsort(cost[prmU, prmV])
        U.permute([prmU[i] for i in reversed(idx)])
        V.permute([prmV[i] for i in reversed(idx)])

    # if permute_U is False, then only permute the factors for V
    elif permute_V:
        V.permute([prmV[i] for i in np.argsort(prmU)])

    # if permute_V is False, then only permute the factors for U
    elif permute_U:
        U.permute([prmU[i] for i in np.argsort(prmV)])

    # else, don't permute anything or flip signs
    else:
        return similarity

    # FLIP SIGNS OF FACTORS
    # ~~~~~~~~~~~~~~~~~~~~~
    flips = np.sign([F[prmU, prmV] for F in sim_matrices])
    flips[0] *= np.prod(flips, axis=0)  # flip an even number of factors
    if permute_U:
        for i, f in enumerate(flips):
            U.factors[i] *= f
    elif permute_V:
        for i, f in enumerate(flips):
            V.factors[i] *= f

    # return the similarity score
    return similarity
