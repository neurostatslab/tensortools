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

def plot_kruskal(factors, figsize=(5,10), lspec='-', plot_n=None, plots='line',
                 titles='', color='b', alpha=1.0, lw=2, dashes=None, sort_fctr=False,
                 link_yaxis=False, label=None, xlabels='', suptitle=None, fig=None,
                 axes=None, yticks=True, width_ratios=None, scatter_kwargs=dict()):
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
        if arg is None or isinstance(arg, argtype):
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
    xlabels = _broadcast_arg(titles, str, 'xlabels')
    lspec = _broadcast_arg(lspec, str, 'lspec')
    color = _broadcast_arg(color, (str,tuple), 'color')
    alpha = _broadcast_arg(alpha, (int,float), 'alpha')
    lw = _broadcast_arg(lw, (int,float), 'lw')
    dashes = _broadcast_arg(dashes, tuple, 'dashes')
    sort_fctr = _broadcast_arg(sort_fctr, (int,float), 'sort_fctr')
    link_yaxis = _broadcast_arg(link_yaxis, (int,float), 'link_yaxis')
    scatter_kwargs = _broadcast_arg(scatter_kwargs, (dict), 'scatter_kwargs')

    # parse plot widths, defaults to equal widths
    if width_ratios is None:
        width_ratios = [1 for _ in range(ndim)]

    # default scatterplot options
    for sckw in scatter_kwargs:
        if not "edgecolor" in sckw.keys():
            sckw["edgecolor"] = "none"
        if not "s" in sckw.keys():
            sckw["s"] = 10

    # setup subplots (unless already specified)
    if fig is None and axes is None:
        fig, axes = plt.subplots(R, ndim,
                               figsize=figsize,
                               gridspec_kw=dict(width_ratios=width_ratios))
    elif fig is None:
        fig = axes[0,0].get_figure()
    else:
        axes = np.array(fig.get_axes(), dtype=object).reshape(R, ndim)

    # check label input
    if label is not None and not isinstance(label, str):
        raise ValueError('label must be None or string')

    # order to plot loadings for each factor
    o = []
    for srt,f in zip(sort_fctr, factors):
        if srt:
            o.append(np.argsort(f[:,0])[::-1])
        else:
            o.append(range(f.shape[0]))

    # main loop, plot each factor
    for r in range(R):
        for i, f in enumerate(factors):

            # determine type of plot
            if plots[i] == 'bar':
                axes[r,i].bar(range(f.shape[0]), f[o[i],r], color=color[i], alpha=alpha[i], label=label)
            elif plots[i] == 'scatter':
                axes[r,i].scatter(range(f.shape[0]), f[o[i],r], c=color[i], alpha=alpha[i], label=label, **scatter_kwargs[i])
            elif plots[i] == 'line':
                ln, = axes[r,i].plot(f[o[i],r], lspec[i], color=color[i], lw=lw[i], alpha=alpha[i], label=label)
                if dashes[i] is not None:
                    ln.set_dashes(dashes[i])
            else:
                raise ValueError('invalid plot type')

            # format axes
            axes[r,i].locator_params(nbins=4)
            axes[r,i].set_xlim([0,f.shape[0]])
            axes[r,i].spines['top'].set_visible(False)
            axes[r,i].spines['right'].set_visible(False)
            axes[r,i].xaxis.set_tick_params(direction='out')
            axes[r,i].yaxis.set_tick_params(direction='out')
            axes[r,i].yaxis.set_ticks_position('left')
            axes[r,i].xaxis.set_ticks_position('bottom')

            # put title on top row
            if r == 0:
                axes[r,i].set_title(titles[i])

            # remove xticks on all but bottom row
            if r != R-1:
                plt.setp(axes[r,i].get_xticklabels(), visible=False)
            else:
                axes[r,i].set_xlabel(xlabels[i])

            # allow user to suppress yticks
            if not yticks:
                axes[r,i].set_yticks([])
            else:
                # only two labels
                yt = axes[r,i].get_yticks()
                ylab = [str(yt[0]), *['' for _ in range(len(yt)-2)], str(yt[-1])]
                axes[r,i].set_yticklabels(ylab)

    # backtrack and fix y-axes to have the same limits
    for i in np.where(link_yaxis)[0]:
        yl = [a.get_ylim() for a in axes[:,i]]
        y0 = min([y[0] for y in yl])
        y1 = max([y[1] for y in yl])
        [a.set_ylim([y0,y1]) for a in axes[:,i]]

    plt.tight_layout()

    return fig, axes

def normalize_kruskal(factors):
    """Normalizes all factors to unit length
    """
    ndim, rank = _validate_kruskal(factors)

    # factor norms
    lam = np.ones(rank)

    # destination for new ktensor
    newfactors = []

    # normalize columns of factor matrices
    lam = np.ones(rank)
    for fact in factors:
        s = np.linalg.norm(fact, axis=0)
        lam *= s
        newfactors.append(fact/(s+1e-20))

    return newfactors, lam

def standardize_kruskal(factors, lam_ratios=None):
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

    # normalize tensor
    nrmfactors, lam = normalize_kruskal(factors)

    # default to equally sized factors
    if lam_ratios is None:
        lam_ratios = np.ones(len(factors))
    
    # check input is valid
    if len(lam_ratios) != len(factors):
        raise ValueError('list of scalings must match the number of tensor modes/dimensions')
    elif np.min(lam_ratios) < 0:
        raise ValueError('list of scalings must be nonnegative')
    else:
        lam_ratios = np.array(lam_ratios) / np.sum(lam_ratios)

    # sort factors by their length/norm and return
    prm = np.argsort(lam)[::-1]
    return [f[:,prm]*np.power(lam[prm], r) for f, r in zip(nrmfactors, lam_ratios)]

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

    A, lamA = normalize_kruskal(A)
    B, lamB = normalize_kruskal(B)

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

    # flip signs in A
    flipped_A = [s*a for s, a in zip(sgn, A)]
    aligned_B = [np.power(l, 1/ndim)*b for l, b in zip(lamB, B)]

    # permute A to align with B
    aligned_A = [a[:,best_perm] for a in flipped_A]
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
