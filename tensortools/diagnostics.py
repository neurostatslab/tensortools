"""
Diagnostic measures for CP decomposition fits
"""

import numpy as np
from copy import deepcopy
from munkres import Munkres
import pdb


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

    # ~~~~~~~~~~~~~~~~~~~~~
    # COMPUTE MATCHING COST
    # ~~~~~~~~~~~~~~~~~~~~~
    unrm = [f / np.linalg.norm(f, axis=0) for f in U.factors]
    vnrm = [f / np.linalg.norm(f, axis=0) for f in V.factors]
    sim_matrices = [np.dot(u.T, v) for u, v in zip(unrm, vnrm)]
    cost = 1 - np.mean(np.abs(sim_matrices), axis=0)

    # solve matching problem via Hungarian algorithm
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # compute similarity across all factors
    similarity = np.mean(1 - cost[prmU, prmV])

    # If U and V are of different ranks, identify unmatched factors.
    unmatched_U = list(set(range(U.rank)) - set(prmU))
    unmatched_V = list(set(range(V.rank)) - set(prmV))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DETERMINE ORDER OF FACTORS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # If permuting both U and V, order factors from most to least similar.
    if permute_U and permute_V:
        idx = np.argsort(cost[prmU, prmV])
    # if permute_U is False, then only permute the factors for V
    elif permute_V:
        idx = np.argsort(prmU)
    # if permute_V is False, then only permute the factors for U
    elif permute_U:
        idx = np.argsort(prmV)
    # else, don't permute anything or flip signs
    else:
        return similarity

    # new permutations
    prmU = [prmU[i] for i in idx]
    prmV = [prmV[i] for i in idx]

    # permute factors
    if permute_U:
        U.permute(prmU)
    if permute_V:
        V.permute(prmV)

    # ~~~~~~~~~~~~~~~~~~~~~
    # FLIP SIGNS OF FACTORS
    # ~~~~~~~~~~~~~~~~~~~~~
    flips = np.sign([F[prmU, prmV] for F in sim_matrices])
    flips[0] *= np.prod(flips, axis=0)  # always flip an even number of factors

    if permute_U:
        for i, f in enumerate(flips):
            U.factors[i] *= f
    elif permute_V:
        for i, f in enumerate(flips):
            V.factors[i] *= f

    # return the similarity score
    return similarity
