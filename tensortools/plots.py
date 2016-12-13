"""
Plotting options for tensor decompositions.
"""

import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from .kruskal import _validate_kruskal
from tensorly.tenalg import norm
from tensorly.kruskal import kruskal_to_tensor
from jetpack import nospines, tickdir

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
    link_yaxis = _broadcast_arg(link_yaxis, (bool), 'link_yaxis')
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

    # link y-axes within columns
    for i in np.where(link_yaxis)[0]:
        yl = [a.get_ylim() for a in axes[:,i]]
        y0 = min([y[0] for y in yl])
        y1 = max([y[1] for y in yl])
        [a.set_ylim([y0,y1]) for a in axes[:,i]]

    # format y-ticks
    for r in range(R):
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

    return fig, axes

def plot_scree(data, models, err=None, ax=None, jitter=0.1,
               scatter_kwargs=dict(edgecolor='none', color='k', alpha=0.6, zorder=2),
               plot_kwargs=dict(color='r', lw=3, zorder=1), **fig_kwargs):
    """Plots relative reconstruction error as a function of model complexity
    """

    if ax is None:
        ax = plt.gca()

    # compute model ranks
    ranks = np.array([_validate_kruskal(model)[1] for model in models], dtype=int)

    # if the user doesn't provide the reconstruction errors, recompute them
    if err is None:
        err = np.array([norm(data - kruskal_to_tensor(model), 2) / norm(data, 2) for model in models])

    # add horizontal jitter 
    n_models = len(models)
    if jitter is not None:
        jit = (np.random.rand(n_models)-0.5)*jitter
    else:
        jit = np.zeros(n_models)

    # plot line connecting minimum reconstruction error for each rank
    unique_ranks = list(set(ranks))
    min_err = [np.min(err[ranks == rank]) for rank in unique_ranks]
    ax.plot(unique_ranks, min_err, **plot_kwargs)

    # plot all fits
    ax.scatter(ranks+jit, err, **scatter_kwargs)

    # format axes
    x0,x1 = min(ranks),max(ranks)
    ax.set_xticks(range(x0,x1+1))
    ax.set_xlim([x0-0.5, x1+0.5])

    nospines(ax=ax)
    tickdir(ax=ax)

    return ax

