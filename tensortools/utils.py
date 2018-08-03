"""
Useful helper functions, not critical to core functionality of tensortools.
"""

import numpy as np
import scipy.spatial
import math
import scipy as sci
from .tensor_utils import unfold


def multilinear_pr(tensor):
    prs = []
    for m in range(tensor.ndim):
        M = unfold(tensor, m)
        lam = np.linalg.svd(M - M.mean(axis=-1, keepdims=True), compute_uv=False, full_matrices=False) ** 2
        prs.append(np.sum(lam)**2 / np.sum(lam**2))
    return prs


def coarse_grain_1d(tensor, factor, axis=0, reducer=np.sum,
                    pad_mode='constant', pad_kwargs=dict(constant_values=0)):
    """Coarse grains a large tensor along axis by factor

    Args
    ----
    tensor : ndarray
    factor : int
        multiplicat
    axis : int
        mode to coarse grain (default=0)
    reducer : function
        reducing function to implement coarse-graining
    """
    if not isinstance(factor, int):
        raise ValueError('coarse-graining factor must be an integer.')
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError('invalid axis for coarse-graining.')

    # compute reshaping dimensions
    new_shape = [s for s in tensor.shape]
    new_shape[axis] = math.ceil(new_shape[axis]/factor)
    new_shape.insert(axis+1, factor)

    # pad tensor if necessary
    pad_width = factor*new_shape[axis] - tensor.shape[axis]
    if pad_width > 0:
        pw = [pad_width if (a == axis) else 0 for a in range(tensor.ndim)]
        tensor = np.pad(tensor, pw, pad_mode, **pad_kwargs)

    # sanity check
    assert pad_width >= 0

    # coarse-grain
    return reducer(tensor.reshape(*new_shape), axis=axis+1)


def coarse_grain(tensor, factors, **kwargs):
    """Coarse grains a large tensor along all modes by specified factors
    """
    for axis, factor in enumerate(factors):
        tensor = coarse_grain_1d(tensor, factor, axis=axis, **kwargs)

    return tensor


def soft_cluster_factor(factor):
    """Returns soft-clustering of data based on CP decomposition results.

    Args
    ----
    factors : ndarray
        Matrix holding low-dimensional CP factors in columns

    Returns
    -------
    cluster_ids : ndarray of ints
        List of cluster assignments for each row of factor matrix
    perm : ndarray of ints
        Permutation that groups rows by clustering and factor magnitude
    """

    # copy factor of interest
    f = np.copy(factor)

    # cluster based on score of maximum absolute value
    cluster_ids = np.argmax(np.abs(f), axis=1)
    scores = f[range(f.shape[0]), cluster_ids]

    # resort within each cluster
    perm = []
    for cluster in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cluster)[0]
        perm += list(idx[np.argsort(scores[idx])][::-1])

    return cluster_ids, perm


def resort_factor_hclust(U):
    """Sorts the rows of a matrix by hierarchical clustering

    Parameters:
        U (ndarray) : matrix of data

    Returns:
        prm (ndarray) : permutation of the rows
    """

    from scipy.cluster import hierarchy
    Z = hierarchy.ward(U)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, U))


def resort_factor_tsp(factor, niter=100000, metric='euclidean', split='dummy', **kwargs):
    """Sorts the factor to (approximately) to solve the traveling
    salesperson problem, so that data elements (rows of factor)
    are placed closed to each other.
    """

    # Compute pairwise distances between all datapoints
    N = factor.shape[0]
    D = scipy.spatial.distance.pdist(factor, metric=metric, **kwargs)

    if split == 'dummy':
        # To solve the traveling salesperson problem with no return to the original node
        # we add a dummy node that has distance zero connections to all other nodes. The
        # dummy node is then removed after we've converged to a solution
        dist = np.zeros((N+1, N+1))
        dist[:N,:N] = scipy.spatial.distance.squareform(D)
    elif split == 'min':
        dist = scipy.spatial.distance.squareform(D)
    else:
        raise ValueError('split parameter not recognized')

    # solve TSP
    path, cost_hist = _solve_tsp(dist, niter)

    if split == 'dummy':
        brk = np.argwhere(path==N).ravel()[0]
        path = np.hstack((path[(brk+1):], path[:brk]))
    elif split == 'min':
        m = np.argmin(np.abs(factor).sum(axis=1))
        brk = np.argwhere(path == m).ravel()[0]
        path = np.hstack((path[brk:], path[:brk]))

    return path, cost_hist


def reverse_segment(path, n1, n2):
    """Reverse the nodes between n1 and n2.
    """
    q = path.copy()
    if n2 > n1:
        q[n1:(n2+1)] = path[n1:(n2+1)][::-1]
        return q
    else:
        seg = np.hstack((path[n1:], path[:(n2+1)]))[::-1]
        brk = len(q) - n1
        q[n1:] = seg[:brk]
        q[:(n2+1)] = seg[brk:]
        return q


def _solve_tsp(dist, niter):
    """Solve travelling salesperson problem (TSP) by two-opt swapping.

    Params
    ------
    dist (ndarray) : distance matrix

    Returns
    -------
    path (ndarray) : permutation of nodes in graph (rows of dist matrix)
    """

    # number of nodes
    N = dist.shape[0]

    # tsp path for quick calculation of cost
    ii = np.arange(N)
    jj = np.hstack((np.arange(1, N), 0))

    # for each node, a sorted list of closest nodes
    dsort = [np.argsort(d) for d in dist]
    dsort = [d[d != i] for i, d in enumerate(dsort)]

    # randomly initialize path through graph
    path = np.random.permutation(N)
    idx = np.argsort(path)
    cost = np.sum(dist[path[ii], path[jj]])

    # keep track of objective function over time
    cost_hist = [cost]

    # optimization loop
    node = 0
    itercount = 0
    n = 0

    while n < N and itercount < niter:

        # count iterations
        itercount += 1

        # we'll try breaking the connection i -> j
        i = path[node]
        j = path[(node+1) % N]

        # We are breaking i -> j so we can remove the cost of that connection.
        c = cost - dist[i, j]

        # Search over nodes k that are closer to j than i.
        for k in dsort[j]:
            # Can safely continue if dist[i,j] < dist[k,j] for the remaining k.
            if k == i:
                n += 1
                break

            # Break connection k -> p.
            # Add connection j -> p.
            # Add connection i -> k.
            p = path[(idx[k]+1) % N]
            new_cost = c - dist[k, p] + dist[j, p] + dist[i, k]

            # If this swap improves the cost, implement it and move to next i.
            if new_cost < cost:
                path = reverse_segment(path, idx[j], idx[k])
                idx = np.argsort(path)
                cost = new_cost
                # Restart from the begining of the graph.
                cost_hist.append(cost)
                n = 0
                break

        # move to next node
        node = (node + 1) % N

    return path, cost_hist
