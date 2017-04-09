"""
Plotting options for tensor decompositions.
"""

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from .kruskal import align_factors, _validate_factors
from tensorly.tenalg import norm
from tensorly.kruskal import kruskal_to_tensor
from jetpack import nospines, tickdir
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
                plot_obj[r,i] = axes[r,i].bar(np.arange(1, f.shape[0]+1), f[:,r], **bar_kw[i])
                axes[r,i].set_xlim(0, f.shape[0]+1)
            elif plots[i] == 'scatter':
                plot_obj[r,i] = axes[r,i].scatter(range(f.shape[0]), f[:,r], **scatter_kw[i])
                axes[r,i].set_xlim(0, f.shape[0])
            elif plots[i] == 'line':
                plot_obj[r,i] = axes[r,i].plot(f[:,r], '-', **line_kw[i])
                axes[r,i].set_xlim(0, f.shape[0])
            else:
                raise ValueError('invalid plot type')

            # format axes
            axes[r,i].locator_params(nbins=4)
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

                # remove decimals from labels
                if ymin.is_integer():
                    ymin = int(ymin)
                if ymax.is_integer():
                    ymax = int(ymax)

                # update plot
                axes[r,i].set_yticks([ymin, ymax])
                axes[r,i].set_yticklabels([str(ymin), str(ymax)])

    plt.tight_layout()

    return fig, axes, plot_obj

def plot_scree(results, ax=None, jitter=0.1, labels=True, scatter_kw=dict(), line_kw=dict()):
    """Plots reconstruction error as a function of model rank.

    Args
    ----
    results : dict
        holds results/output of `cpd_batch_fit`
    ax : matplotlib axis (optional)
        axis to plot on (defaults to current axis object)
    jitter : float (optional)
        amount of horizontal jitter added to scatterpoints (default=0.1)
    labels : bool (optional)
        if True, label the x and y axes (default=True)
    scatter_kw : dict (optional)
        keyword arguments for styling the scatterpoints
    line_kw : dict (optional)
        keyword arguments for styling the line
    """

    if ax is None:
        ax = plt.gca()

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
    ax.scatter(ranks_jit.ravel(), err, **scatter_kw)
    ax.plot(ranks[:, 0], min_err, **line_kw)

    if labels:
        ax.set_xlabel('model rank')
        ax.set_ylabel('Norm of resids / Norm of data')
    
    return ax

def plot_similarity(results, ax=None, jitter=0.1, labels=True, scatter_kw=dict(), line_kw=dict()):
    """Plots model similarity as a function of model rank
    
    Args
    ----
    results : dict
        holds results/output of `cpd_batch_fit`
    ax : matplotlib axis (optional)
        axis to plot on (defaults to current axis object)
    jitter : float (optional)
        amount of horizontal jitter added to scatterpoints (default=0.1)
    labels : bool (optional)
        if True, label the x and y axes (default=True)
    scatter_kw : dict (optional)
        keyword arguments for styling the scatterpoints
    line_kw : dict (optional)
        keyword arguments for styling the line
    """
    
    if ax is None:
        ax = plt.gca()

    # compile statistics for plotting
    ranks, sim, mean_sim = [], [], []
    for r in results.keys():
        ranks.append([r for _ in range(len(results[r]['factors'])-1)])
        sim += list(results[r]['similarity'][1:])
        mean_sim.append(np.mean(results[r]['similarity'][1:]))

    # add horizontal jitter 
    ranks = np.array(ranks)
    if jitter is not None:
        ranks_jit = ranks + (np.random.rand(*ranks.shape)-0.5)*jitter

    # make plot
    ax.scatter(ranks_jit.ravel(), sim, **scatter_kw)
    ax.plot(ranks[:, 0], mean_sim, **line_kw)

    if labels:
        ax.set_xlabel('model rank')
        ax.set_ylabel('Norm of resids / Norm of data')
        
    ax.scatter(ranks_jit.ravel(), sim, **scatter_kw)

    # axis labels
    if labels:
        ax.set_xlabel('model rank')
        ax.set_ylabel('model similarity')
    
    return ax

def plot_sim_v_err(results, axes=None, fig=None, figsize=None, labels=True, sharex=True,
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
            ax.set_ylim(0,1)

        ax = axes.ravel()[0]
        xl = [0, np.round(ax.get_xlim()[1], 2)]
        ax.set_xlim(xl)

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
