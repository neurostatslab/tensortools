"""
Useful helper functions, not critical to core functionality of tensortools.
"""

import numpy as np
import scipy
import math


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
        pw = [pad_width if a==axis else 0 for a in range(tensor.ndim)]
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

    # resort based on cluster assignment
    #i0 = np.argsort(cluster_ids)
    #f, scores = f[i0], scores[i0]

    # resort within each cluster
    perm = []
    for cluster in np.unique(cluster_ids):
        idx = np.where(cluster_ids == cluster)[0]
        perm += list(idx[np.argsort(scores[idx])][::-1])

    return cluster_ids, perm

def resort_factor_tsp(factor, niter=1000, metric='euclidean', **kwargs):
    """Sorts the factor to (approximately) to solve the traveling
    salesperson problem, so that data elements (rows of factor)
    are placed closed to each other.
    """

    # Compute pairwise distances between all datapoints
    N = factor.shape[0]
    D = scipy.spatial.distance.pdist(factor, metric=metric, **kwargs)
    
    # To solve the travelling salesperson problem with no return to the original node
    # we add a dummy node that has distance zero connections to all other nodes. The
    # dummy node is then removed after we've converged to a solution
    dist = np.zeros((N+1, N+1))
    dist[:N,:N] = scipy.spatial.distance.squareform(D)
    
    # solve TSP
    path, cost_hist = solve_tsp(D)
    
    # remove dummy node at position i
    i = np.argwhere(path==N+1).ravel()[0]
    path = np.hstack((path[(i+1):], path[:i]))
    
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

def solve_tsp(dist):
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
    while node < N:

        # we'll try breaking the connection i -> j
        i = path[node]
        j = path[(node+1) % N]
        
        # since we are breaking i -> j we can remove the cost of that connection
        c = cost - dist[i, j]

        # search over nodes k that are closer to j than i
        for k in dsort[j]:
            # can safely continue if dist[i,j] < dist[k,j] for the remaining k
            if k == i:
                node += 1
                break

            # break connection k -> p
            # add connection j -> p
            # add connection i -> k
            p = path[(idx[k]+1) % N]
            new_cost = c - dist[k,p] + dist[j,p] + dist[i,k]

            # if this swap improves the cost, implement it and move to next i
            if new_cost < cost:
                path = reverse_segment(path, idx[j], idx[k])
                idx = np.argsort(path)
                # make sure that we didn't screw up
                assert np.abs(np.sum(dist[path[ii], path[jj]]) - new_cost) < 1e-6
                cost = new_cost
                # restart from the begining of the graph
                cost_hist.append(cost)
                node = 0
                break

    return path, cost_hist

