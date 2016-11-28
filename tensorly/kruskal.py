"""
Core operations on Kruskal tensors.
"""

import numpy as np
from .base import fold, tensor_to_vec
from .tenalg import khatri_rao
from matplotlib import gridspec
import matplotlib.pyplot as plt
import itertools as itr

# Author: Jean Kossaifi

# License: BSD 3 clause


def kruskal_to_tensor(factors):
    """Turns the Khatri-product of matrices into a full tensor

        ``factor_matrices = [|U_1, ... U_n|]`` becomes
        a tensor shape ``(U[1].shape[0], U[2].shape[0], ... U[-1].shape[0])``

    Parameters
    ----------
    factors : ndarray list
        list of factor matrices, all with the same number of columns
        i.e. for all matrix U in factor_matrices:
        U has shape ``(s_i, R)``, where R is fixed and s_i varies with i

    Returns
    -------
    ndarray
        full tensor of shape ``(U[1].shape[0], ... U[-1].shape[0])``

    Notes
    -----
    This version works by first computing the mode-0 unfolding of the tensor
    and then refolding it.
    There are other possible and equivalent alternate implementation.

    Version slower but closer to the mathematical definition
    of a tensor decomposition:

    >>> from functools import reduce
    >>> def kt_to_tensor(factors):
    ...     for r in range(factors[0].shape[1]):
    ...         vecs = np.ix_(*[u[:, r] for u in factors])
    ...         if r:
    ...             res += reduce(np.multiply, vecs)
    ...         else:
    ...             res = reduce(np.multiply, vecs)
    ...     return res

    """
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)


def kruskal_to_unfolded(factors, mode):
    """Turns the khatri-product of matrices into an unfolded tensor

        turns ``factors = [|U_1, ... U_n|]`` into a mode-`mode`
        unfolding of the tensor

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
    ndarray
        unfolded tensor of shape (tensor_shape[mode], -1)

    Notes
    -----
    Writing factors = [U_1, ..., U_n], we exploit the fact that
    ``U_k = U[k].dot(khatri_rao(U_1, ..., U_k-1, U_k+1, ..., U_n))``
    """
    return factors[mode].dot(khatri_rao(factors, skip_matrix=mode).T)


def kruskal_to_vec(factors):
    """Turns the khatri-product of matrices into a vector

        (the tensor ``factors = [|U_1, ... U_n|]``
        is converted into a raveled mode-0 unfolding)

    Parameters
    ----------
    factors : ndarray list
        list of matrices, all with the same number of columns
        i.e.::

            for u in U:
                u[i].shape == (s_i, R)
                
        where `R` is fixed while `s_i` can vary with `i`

    Returns
    -------
    ndarray
        vectorised tensor
    """
    return tensor_to_vec(kruskal_to_tensor(factors))

def plot_kruskal(factors, lspec='-', plot_n=None, plots='line', titles='',
                 color='b', lw=2, sort_fctr=False, link_yaxis=False, label=None,
                 xlabels='', gs=None, yticks=True, width_ratios=None):
    """Plots a KTensor.

    Each parameter can be passed as a list if different formatting is
    desired for each set of factors. For example, if `X` is a 3rd-order
    tensor (i.e. `X.ndim == 3`) then `X.plot(color=['r','k','b'])` plots
    all factors for the first mode in red, the second in black, and the
    third in blue. On the other hand, `X.plot(color='r')` produces red
    plots for each mode.

    Parameters
    ----------
    lspec : str or list
        Matplotlib linespec (e.g. '-' or '--'). Default is '-'.
    plot_n : int or list
        Number of factors to plot. The default is to plot all factors.
    plots : str or list
        One of {'bar','line'} to specify the type of plot for each factor.
        The default is 'line'.
    titles : str or list
        Plot title for each set of factors
    color : matplotlib color or list
        Color for plots associated with each set of factors
    lw : int or list
        Specifies line width on plots. Default is 2
    sort_fctr : bool or list
        If true, sorts each factor before plotting. This is useful for
        modes of the tensor that have no natural ordering. Default is
        False.
    link_yaxis : bool or list
        If True, set ylim to the same extent for each set of factors.
        Default is True.
    """

    ndim, rank = _validate_kruskal(factors)

    # helper function for parsing plot options
    def _broadcast_arg(arg, argtype, name):
        """Broadcasts plotting option `arg` to all factors
        """
        if isinstance(arg, argtype):
            return [arg for _ in range(ndim)]
        elif isinstance(arg, list):
            return arg
        else:
            raise ValueError('Parameter %s must be a %s or a list'
                             'of %s' % (name, argtype, argtype))

    # dimensionality of tensor and number of factors to plot
    R = rank if plot_n is None else plot_n

    # parse optional inputs
    plots = _broadcast_arg(plots, str, 'plots')
    titles = _broadcast_arg(titles, str, 'titles')
    lspec = _broadcast_arg(lspec, str, 'lspec')
    color = _broadcast_arg(color, (str,tuple), 'color')
    lw = _broadcast_arg(lw, (int,float), 'lw')
    sort_fctr = _broadcast_arg(sort_fctr, (int,float), 'sort_fctr')
    link_yaxis = _broadcast_arg(link_yaxis, (int,float), 'link_yaxis')

    # parse plot widths, defaults to equal widths
    if width_ratios is None:
        width_ratios = [1 for _ in range(ndim)]

    # setup subplots (unless gridspec already specified)
    if gs is None:
        gs = gridspec.GridSpec(R, ndim, width_ratios=width_ratios)

    # check label input
    if label is not None and not isinstance(label, str):
        raise ValueError('label must be None or string')

    # order to plot loadings for each factor
    o = []
    for srt,f in zip(sort_fctr, factors):
        if srt:
            o.append(np.argsort(f[:,0]))
        else:
            o.append(range(f.shape[0]))

    # main loop, plot each factor
    s = 0   # subplot counter
    for r in range(R):
        for i, f in enumerate(factors):
            plt.subplot(gs[s])

            # determine type of plot
            if plots[i] == 'bar':
                plt.bar(range(f.shape[0]), f[o[i],r], label=label)
            elif plots[i] == 'scatter':
                plt.scatter(range(f.shape[0]), f[o[i],r], c=color[i], label=label)
            elif plots[i] == 'line':
                plt.plot(f[o[i],r], lspec[i], color=color[i], lw=lw[i], label=label)
            else:
                raise ValueError('invalid plot type')

            # format axes
            plt.locator_params(nbins=4)
            plt.xlim([0,f.shape[0]])

            # put title on top row
            if r == 0:
                plt.title(titles[i])

            # remove xticks on all but bottom row
            if r != R-1:
                plt.xticks([])
                plt.xlabel(xlabels[i])

            # allow user to suppress yticks
            if not yticks:
                plt.yticks([])

            # move to next subplot
            s += 1

    # backtrack and fix y-axes to have the same limits
    if any(link_yaxis):
        # determine y limits
        s = 1
        yls = np.empty((R, ndim, 2))
        for r, n in itr.product(range(R), range(ndim)):
            plt.subplot(R, ndim, s)
            yls[r,n,:] = plt.ylim()
            s += 1
        y0 = np.amin(yls[:,:,0],axis=0)
        y1 = np.amax(yls[:,:,1],axis=0)

        # set y limits
        s = 1
        for r, n in itr.product(range(R), range(ndim)):
            if link_yaxis[n]:
                plt.subplot(R, ndim, s)
                plt.ylim([y0[n],y1[n]])
            s += 1

    return gs
    
def standardize_kruskal(factors):
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
    lam : 1darray
        norm of each factor
    """

    ndim, rank = _validate_kruskal(factors)

    # factor norms
    lam = np.ones(rank)

    # allocate normalized factors
    nrm_factors = [np.empty(fact.shape) for fact in factors]

    # normalize factors to unit length
    for fact, n_fact in zip(factors, nrm_factors):
        
        for r in range(rank):
            # normalizing constant (prevent div-by-zero)
            s = np.linalg.norm(fact[:,r])

            # prevent div-by-zero
            if s < 1e-20: s += 1e-20

            # normalize to unit length
            n_fact[:,r] = fact[:,r] / s
            lam[r] *= s

    # sort factors by their length/norm and return
    prm = np.argsort(lam)[::-1]
    return [n_fact[:,prm] for n_fact in nrm_factors], lam[prm]

def redistribute_kruskal(factors, ratios=None):
    """Rescales factor matrices relative to ratios parameter
    """

    ndim, rank = _validate_kruskal(factors)

    # check inputs
    if ratios is None:
        ratios = np.ones(ndim)

    # convert to numpy array
    if len(ratios) != ndim:
        raise ValueError('list of scalings must match the number of tensor modes/dimensions')
    elif np.min(ratios) < 0:
        raise ValueError('list of scalings must be nonnegative')
    else:
        ratios = np.array(ratios) / np.sum(ratios)

    # destination for new ktensor
    newfactors = []

    # normalize columns of factor matrices
    lam = np.ones(rank)
    for fact in factors:
        s = np.linalg.norm(fact, axis=0)
        lam *= s
        newfactors.append(fact/(s+1e-20))

    # redistribute lambdas amongst factor matrices
    for r, fact in zip(ratios, newfactors):
        fact *= np.power(lam, r)

    return newfactors


def align_kruskal(A, B, greedy=True, penalize_lam=True):
    """Align two kruskal tensors

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
    lam : 1darray
        norm of each factor
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
    ra = A[0].shape[1]
    rb = B[0].shape[1]

    if ra < rb:
        raise ValueError('tensor A must have at least as many components as tensor B.')

    A, lamA = standardize_kruskal(A)
    B, lamB = standardize_kruskal(B)

    # compute dot product
    dprod = np.array([np.dot(a.T, b) for a, b in zip(A, B)])

    # similarity matrix
    sim = np.multiply.reduce([np.abs(dp) for dp in dprod])

    # include penalty on factor lengths
    if penalize_lam:
        for i, j in itr.product(range(ra), range(rb)):
            la, lb = lamA[i], lamB[j]
            sim[i, j] *= 1 - (abs(la-lb) / max(abs(la),abs(lb)))

    if greedy:
        # find permutation of factors by a greedy method
        best_perm = -np.ones(ra, dtype='int')
        score = 0
        for r in range(rb):
            i, j = np.unravel_index(np.argmax(sim), sim.shape)
            score += sim[i,j]
            sim[i,:] = -1
            sim[:,j] = -1
            best_perm[j] = i
        score /= rb

    else:
        # search all permutations
        score = 0
        for comb in itr.combinations(range(ra), rb):
            for perm in itr.permutations(comb):
                sc = sum([ sim[i,j] for j, i in enumerate(perm)])
                if sc > score:
                    best_perm = np.array(perm)
                    score = sc
        score /= rb

    # if ra > rb, fill in permutation with remaining factors
    unset = list(set(range(ra)) - set(best_perm))
    best_perm[unset] = range(rb, ra)

    # Flip signs of ktensor factors for better alignment
    sgn = np.tile(np.power(lamA, 1/ndim), (ndim,1))
    for j in range(rb):

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

    # flip and permute to align
    aligned_A = [s*a[:,best_perm] for s, a in zip(sgn, A)]
    aligned_B = [np.power(l, 1/ndim)*b for l, b in zip(lamB, B)]

    return aligned_A, aligned_B, score

def _validate_kruskal(factors):
    """Checks that input is a valid kruskal tensor

    Returns
    -------
    ndim : int
        number of dimensions in tensor
    rank : int
        number of factors
    """
    ndim = len(factors)
    rank = factors[0].shape[1]

    for f in factors:
        if f.shape[1] != rank:
            raise ValueError('KTensor has inconsistent rank along modes.')

    return ndim, rank
