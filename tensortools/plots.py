"""
Plotting options for tensor decompositions.
"""

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from .kruskal import align_factors, _validate_factors
from tensorly.tenalg import norm
from tensorly.kruskal import kruskal_to_tensor
from jetpack import nospines, tickdir, breathe, minlabels
import itertools as itr
from sklearn.linear_model import LogisticRegression

def plot_factors(factors, figsize=None, plots='line', ylim='link', fig=None, axes=None,
                 yticks=True, width_ratios=None, scatter_kw=dict(), line_kw=dict(), bar_kw=dict()):
    """Plots a KTensor.

    Each parameter can be passed as a list if different formatting is
    desired for each set of factors. For example, if `X` is a 3rd-order
    tensor (i.e. `X.ndim == 3`) then `X.plot(color=['r','k','b'])` plots
    all factors for the first mode in red, the second in black, and the
    third in blue. On the other hand, `X.plot(color='r')` produces red
    plots for each mode.

    Parameters
    ----------
    plots : str or list
        One of {'bar','line'} to specify the type of plot for each factor.
        The default is 'line'.
    color : matplotlib color or list
        Color for plots associated with each set of factors
    lw : int or list
        Specifies line width on plots. Default is 2
    ylim : str, y-axis limits or list
        Specifies how to set the y-axis limits for each mode of the
        decomposition. For a third-order, rank-2 model, setting
        ylim=['link', (0,1), ((0,1), (-1,1))] specifies that the
        first pair of factors have the same y-axis limits (chosen
        automatically), the second pair of factors both have y-limits
        (0,1), and the third pair of factors have y-limits (0,1) and
        (-1,1).
    """

    factors, ndim, rank = _validate_factors(factors)

    if figsize is None:
        figsize = (8, rank)

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

    # parse optional inputs
    plots = _broadcast_arg(plots, str, 'plots')
    ylim = _broadcast_arg(ylim, (tuple, str), 'ylim')
    bar_kw = _broadcast_arg(bar_kw, dict, 'bar_kw')
    line_kw = _broadcast_arg(line_kw, dict, 'line_kw')
    scatter_kw = _broadcast_arg(scatter_kw, dict, 'scatter_kw')

    # parse plot widths, defaults to equal widths
    if width_ratios is None:
        width_ratios = [1 for _ in range(ndim)]

    # default scatterplot options
    for sckw in scatter_kw:
        if not "edgecolor" in sckw.keys():
            sckw["edgecolor"] = "none"
        if not "s" in sckw.keys():
            sckw["s"] = 10

    # setup subplots (unless already specified)
    if fig is None and axes is None:
        fig, axes = plt.subplots(rank, ndim,
                               figsize=figsize,
                               gridspec_kw=dict(width_ratios=width_ratios))
        if rank == 1: axes = axes[None, :]
    elif fig is None:
        fig = axes[0,0].get_figure()
    else:
        axes = np.array(fig.get_axes(), dtype=object).reshape(rank, ndim)

    # main loop, plot each factor
    plot_obj = np.empty((rank, ndim), dtype=object)
    for r in range(rank):
        for i, f in enumerate(factors):

            # determine type of plot
            if plots[i] == 'bar':
                plot_obj[r,i] = axes[r,i].bar(range(f.shape[0]), f[:,r], **bar_kw[i])
            elif plots[i] == 'scatter':
                plot_obj[r,i] = axes[r,i].scatter(range(f.shape[0]), f[:,r], **scatter_kw[i])
            elif plots[i] == 'line':
                plot_obj[r,i] = axes[r,i].plot(f[:,r], '-', **line_kw[i])
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

            # remove xticks on all but bottom row
            if r != rank-1:
                plt.setp(axes[r,i].get_xticklabels(), visible=False)

    # link y-axes within columns
    for i, yl in enumerate(ylim):
        if yl is None:
            continue
        elif yl == 'link':
            yl = [a.get_ylim() for a in axes[:,i]]
            y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
            [a.set_ylim((y0, y1)) for a in axes[:,i]]
        elif yl == 'tight':
            [a.set_ylim(np.min(factors[i][:,r]), np.max(factors[i][:,r]))  for r, a in enumerate(axes[:,i])]
        elif isinstance(yl[0], (int, float)) and len(yl) == 2:
            [a.set_ylim(yl) for a in axes[:,i]]
        elif isinstance(yl[0], (tuple, list)) and len(yl) == rank:
            [a.set_ylim(lims) for a, lims in zip(axes[:,i], yl)]
        else:
            raise ValueError('ylimits not properly specified')

    # format y-ticks
    for r in range(rank):
        for i in range(ndim):
            if not yticks:
                axes[r,i].set_yticks([])
            else:
                # only two labels
                ymin, ymax = np.round(axes[r,i].get_ylim(), 2)
                axes[r,i].set_ylim((ymin, ymax))

                # reset tick marks
                yt = np.linspace(ymin, ymax, 4)

                # remove decimals from labels
                if ymin.is_integer():
                    ymin = int(ymin)
                if ymax.is_integer():
                    ymax = int(ymax)

                # update plot
                ylab = [str(ymin), *['' for _ in range(len(yt)-2)], str(ymax)]
                axes[r,i].set_yticks(yt)
                axes[r,i].set_yticklabels(ylab)

    plt.tight_layout()

    return fig, axes, plot_obj

def plot_scree(results, yvals=None, axes=None, fig=None, figsize=(6,3), jitter=0.1, labels=True,
               greedy=None,
               scatter_kw=dict(edgecolor='none', color='k', alpha=0.6, zorder=2),
               line_kw=dict(color='r', lw=3, zorder=1)):
    """Plots reconstruction error and model similarity
    """

    # setup figure and axes
    if fig is None and axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    elif fig is None or axes is None:
        raise ValueError('If either fig or axes are given as keyword arguments, both must be specifed.')

    # compile statistics for plotting
    ranks, err, sim, min_err = [], [], [], []
    for r in results.keys():
        # reconstruction errors for rank-r models
        e = list(results[r]['err_final'])
        err += e
        min_err.append(min(e))
        ranks.append([r for _ in range(len(e))])

        # similarity of fitted models
        sim += list(results[r]['similarity'][1:])

    # add horizontal jitter 
    ranks = np.array(ranks)
    if jitter is not None:
        ranks_jit = ranks + (np.random.rand(*ranks.shape)-0.5)*jitter

    # Scree plot
    axes[0].plot(ranks[:, 0], min_err, **line_kw)
    axes[0].scatter(ranks_jit.ravel(), err, **scatter_kw)

    x0, x1 = np.min(ranks), np.max(ranks)
    axes[0].set_xticks(range(x0, x1+1))
    axes[0].set_xlim([x0-0.5, x1+0.5])

    if labels:
        axes[0].set_xlabel('model rank')
        axes[0].set_ylabel('Norm of resids / Norm of data')

    nospines(ax=axes[0])
    tickdir(ax=axes[0])

    # Similarity plot
    axes[1].scatter(ranks_jit[:, 1:].ravel(), sim, **scatter_kw)

    axes[1].set_xticks(range(x0, x1+1))
    axes[1].set_xlim([x0-0.5, x1+0.5])
    axes[1].set_ylim([0, 1.1])
    axes[1].set_yticks([0, 1])
    nospines(ax=axes[1])
    tickdir(ax=axes[1])
    axes[1].spines['left'].set_bounds(0, 1)

    # axis labels
    if labels:
        axes[1].set_xlabel('model rank')
        axes[1].set_ylabel('model similarity')
    
    return fig, axes


def plot_similarity(results, axes=None, fig=None, figsize=None, labels=True, sharex=True,
                    sharey=True, format_axes=True, scatter_kw=dict(edgecolor='none', color='k')):
    """Plots reconstruction error vs model similarity
    """

    # setup figure and axes
    if fig is None and axes is None:
        m, n = _choose_subplots(len(results))
        if figsize is None:
            figsize = (2*n, 2*m)
        fig, axes = plt.subplots(m, n, figsize=figsize, sharex=sharex, sharey=sharey)
    elif fig is None or axes is None:
        raise ValueError('If either fig or axes are given as keyword arguments, both must be specifed.')

    # compile statistics for plotting
    ranks = np.sort(list(results.keys()))
    for r, ax in zip(ranks, axes.ravel()):
        err_diff = results[r]['err_final'][1:] - results[r]['err_final'][0]
        ax.scatter(err_diff, results[r]['similarity'][1:], **scatter_kw)

    if format_axes:
        for ax in axes.ravel():
            nospines(ax=ax)
            tickdir(ax=ax)
            ax.set_ylim(0,1)

        ax = axes.ravel()[0]
        xl = [0, ax.get_xlim()[1]]
        ax.set_xlim(xl)
        breathe(ax=axes.ravel()[0])

        for ax in axes.ravel():
            ax.spines['left'].set_bounds(0,1)
            ax.spines['bottom'].set_bounds(*xl)
        for ax in axes[:,0]:
            ax.set_ylabel('similarity')
        for ax in axes[-1,:]:
            ax.set_xlabel('$\Delta$ error')
        for ax in axes.ravel():
            ax.set_xticks(np.linspace(*xl, 4))
            ax.set_xticklabels([xl[0], '', '', xl[1]])

    return fig, axes

def plot_decode(factors, y, ax=None, lw=3, label=None, Decoder=LogisticRegression, **kwargs):
    """Plot decoding accruacy of metadata for a series of models
    """

    if ax is None:
        ax = plt.gca()

    # extract model ranks
    ranks = np.array([X.shape[1] for X in factors], dtype=int)

    # fit decoders for each set of factors, save accuracy
    scores = []
    for X in factors:
        scores.append(Decoder(**kwargs).fit(X, y).score(X, y))
    scores = np.array(scores)

    # mean accuracy for each rank
    unique_ranks = np.unique(ranks)
    mean_scores = [np.mean(scores[ranks == rank]) for rank in unique_ranks]

    # make plots
    ln, = ax.plot(unique_ranks, mean_scores, lw=lw, label=label)
    dt = ax.scatter(ranks, scores, edgecolor='none', color=ln.get_c(), alpha=0.5)

    nospines(ax=ax)
    tickdir(ax=ax)

    return ln, dt

def plot_persistence(models, ref_rank, ax=None, jitter=0.3, plot_kwargs=dict(alpha=0.5), **kwargs):
    """Plot the persistence score for each factor across all fits.
    """

    if ax is None:
        ax = plt.gca()

    # list of ranks for all models
    model_ranks = [_validate_factors(model)[2] for model in models]

    # list of ranks for each factor
    factor_ranks = []
    factor_scores = []

    for m in range(len(models)):

        # all models, except model m
        other_models = models.copy()
        other_ranks = model_ranks.copy()

        # inspect model m
        model = other_models.pop(m)
        rank = other_ranks.pop(m)

        if rank != ref_rank:
            continue
        print('.')
        sc = np.array([align_factors(om, model, **kwargs)[2] for om in other_models])

        other_ranks = np.array(other_ranks)
        ln, = ax.plot(other_ranks, sc, 'o', alpha=0.3)

        unique_r = np.unique(other_ranks)
        mean_sc = np.array([np.mean(sc[other_ranks == r]) for r in unique_r])
        ax.plot(unique_r, mean_sc, '-', color=ln.get_color(), lw=2)

    ax.set_xlim(np.min(model_ranks)-0.5, np.max(model_ranks)+0.5)
    ax.set_xticks(np.unique(model_ranks))
    nospines(ax=ax)
    tickdir(ax=ax)

    return ax

def _primes(n):
    """Computes the prime factorization for an integer
    """
    pfac = [] # prime factors
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            pfac.append(d)
            n //= d
        d += 1
    if n > 1:
       pfac.append(n)
    return pfac

def _isprime(n):
    """Returns whether an integer is prime or not.
    """
    return all([ n % d != 0 for d in range(2, n//2+1)])

def _choose_subplots(n, tol=2.5):
    """Calculate roughly square layout for subplots
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError('number of subplots must be specified as a positive integer')

    if n == 1:
        return (1, 1)
    
    while _isprime(n) and n > 4:
        n = n+1

    p = _primes(n)

    # single row of plots
    if len(p) == 1:
        return 1, p[0]

    while len(p) > 2:
        if len(p) >= 4:
            p[1] = p[1]*p.pop()
            p[0] = p[0]*p.pop()
        else:
            # len(p) == 3
            p[0] = p[0]*p[1]
            p.pop(1)
        p = np.sort(p)

    if p[1]/p[0] > tol:
        return _choose_subplots(n+1, tol=tol)
    else:
        return p[0], p[1]
