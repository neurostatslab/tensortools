"""
Diagnostic measures for CP decomposition fits
"""

import numpy as np
from copy import deepcopy
from scipy.spatial.distance import cdist
from munkres import Munkres

def kruskal_similarity(U, V):
    """Returns similarity score between zero and one for two kruskal tensors

    Similarity score is based on the average angle between factors
    """
    return kruskal_align(U, V, inplace=False)[-1]

def kruskal_align(U, V, inplace=False):
    """Permutes two KTensors to best align their factors

    Parameters
    ----------
    U : Ktensor
    V : Ktensor
    inplace : bool
        If True, overwrite inputs otherwise create copies (default: False).

    Returns
    -------
    U1 
    """

    # copy U and V so that original results are not overwritten
    if not inplace:
        U, V = deepcopy(U), deepcopy(V)

    # matching cost from U to V
    cost = np.mean([cdist(u.T, v.T, metric='cosine') for u, v in zip(U, V)], axis=0)

    # solve matching problem via Hungarian algorithm
    indices = Munkres().compute(cost)
    prmU, prmV = zip(*indices)

    # If U and V are of different ranks, add unmatched factors to end of permutation
    if U.rank > V.rank:
        prmU += list(set(range(U.rank)) - set(prmU))
    elif U.rank < V.rank:
        prmV += list(set(range(V.rank)) - set(prmV))

    # compute similarity across all factors
    similarity = np.mean(1 - cost[prmU, prmV])

    return U.permute(prmU), V.permute(prmV), similarity

