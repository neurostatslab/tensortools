"""
Diagnostic measures for CP decomposition fits.
"""

import numpy as np
from munkres import Munkres
from scipy.spatial.distance import cdist


def shifted_align(U, V, permute_U=False, permute_V=False):
    """Aligns two models and returns a similarity score.

    Parameters
    ----------
    U : ShiftedCP or MultiShiftModel
        First shifted decomposition to align.
    V : ShiftedCP or MultiShiftModel
        Second shifed decomposition to align.
    permute_U : bool
        If True, modifies 'U' to align the models (default is False).
    permute_V : bool
        If True, modifies 'V' to align the models (default is False).

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

    # Initial model ranks.
    U_init_rank, V_init_rank = U.rank, V.rank

    # Drop any factors with zero magnitude.
    U.prune_()
    V.prune_()

    # Munkres expects V_rank <= U_rank.
    if U.rank > V.rank:
        U.pad_zeros_(U_init_rank - U.rank)
        V.pad_zeros_(V_init_rank - V.rank)
        return kruskal_align(
            V, U, permute_U=permute_V, permute_V=permute_U)

    # Compute per-component estimates for model U.
    Uest = np.empty((U.rank, U.size))
    for r in range(U.rank):
        sds = np.setdiff1d(np.arange(U.rank), r)
        pred = U.predict(skip_dims=sds)
        Uest[r] = pred.ravel()

    # Compute per-component estimates for model V.
    Vest = np.empty((V.rank, V.size))
    for r in range(V.rank):
        sds = np.setdiff1d(np.arange(V.rank), r)
        pred = V.predict(skip_dims=sds)
        Vest[r] = pred.ravel()

    # Compute distances between all model components.
    cost = cdist(Uest, Vest, metric="euclidean") / np.maximum(
        np.linalg.norm(Uest, axis=1)[:, None],
        np.linalg.norm(Vest, axis=1)[None, :]
    )

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

    # Permute the factors and shifts.
    if permute_U:
        U.permute(prmU)
    if permute_V:
        V.permute(prmV)

    # Pad zero factors to restore original ranks.
    U.pad_zeros_(U_init_rank - U.rank)
    V.pad_zeros_(V_init_rank - V.rank)

    # Return the similarity score
    return similarity
