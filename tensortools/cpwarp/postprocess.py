"""
Diagnostic measures for CP decomposition fits.
"""

import numpy as np
from munkres import Munkres

from tensortools.cpwarp import padded_shifts

from tensortools.cpwarp.shifted_cp import ShiftedCP
from tensortools.cpwarp.multishift import MultiShiftModel

from scipy.spatial.distance import cdist


def shifted_align(U, V, permute_U=False, permute_V=False):
    """Aligns two models and returns a similarity score.

    Parameters
    ----------
    U : ShiftedCP
        First kruskal tensor to align.
    V : ShiftedCP
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

    K, T, N = U.shape

    Uest = np.empty((U.rank, K * T * N))
    Vest = np.empty((V.rank, K * T * N))

    for r in range(U.rank):
        pred = U.predict(skip_dims=np.setdiff1d(np.arange(U.rank), r))
        Uest[r] = pred.ravel()

    for r in range(V.rank):
        pred = V.predict(skip_dims=np.setdiff1d(np.arange(V.rank), r))
        Vest[r] = pred.ravel()

    cost = cdist(Uest, Vest, metric="correlation")
    indices = Munkres().compute(cost.copy())
    prmU, prmV = zip(*indices)

    # Compute mean factor similarity given the optimal matching.
    similarity = np.mean(-cost[prmU, prmV])

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
        print(prmU)
        U.permute(prmU)
    if permute_V:
        V.permute(prmV)

    # Return the similarity score
    return similarity
