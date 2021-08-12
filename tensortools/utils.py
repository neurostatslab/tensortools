"""
Miscellaneous functions for interpreting low-dimensional models and data.
"""

import numpy as np
import scipy.spatial


def soft_cluster_factor(factor):
    """Returns soft-clustering of data based on CP decomposition results.

    Parameters
    ----------
    data : ndarray, N x R matrix of nonnegative data
        Datapoints are held in rows, features are held in columns

    Returns
    -------
    cluster_ids : ndarray, vector of N integers in range(0, R)
        List of soft cluster assignments for each row of data matrix
    perm : ndarray, vector of N integers
        Permutation / ordering of the rows of data induced by the soft
        clustering.
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


def tsp_linearize(data, niter=1000, metric='euclidean', **kwargs):
    """Sorts a matrix dataset to (approximately) solve the traveling
    salesperson problem. The matrix can be re-sorted so that sequential rows
    represent datapoints that are close to each other based on some
    user-defined distance metric. Uses 2-opt local search algorithm.

    Args
    ----
    data : ndarray, N x R matrix of data
        Datapoints are held in rows, features are held in columns

    Returns
    -------
    perm : ndarray, vector of N integers
        Permutation / ordering of the rows of data that approximately
        solves the travelling salesperson problem.
    """

    # Compute pairwise distances between all datapoints
    N = data.shape[0]
    D = scipy.spatial.distance.pdist(data, metric=metric, **kwargs)

    # To solve the travelling salesperson problem with no return to the
    # original node we add a dummy node that has distance zero connections
    # to all other nodes. The dummy node is then removed after we've converged
    # to a solution.
    dist = np.zeros((N+1, N+1))
    dist[:N, :N] = scipy.spatial.distance.squareform(D)

    # solve TSP
    perm, cost_hist = _solve_tsp(dist)

    # remove dummy node at position i
    i = np.argwhere(perm == N).ravel()[0]
    perm = np.hstack((perm[(i+1):], perm[:i]))

    return perm


def hclust_linearize(U):
    """Sorts the rows of a matrix by hierarchical clustering.

    Parameters:
        U (ndarray) : matrix of data

    Returns:
        prm (ndarray) : permutation of the rows
    """

    from scipy.cluster import hierarchy
    Z = hierarchy.ward(U)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, U))


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

    # for each node, cache a sorted list of all other nodes in order of
    # increasing distance.
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
