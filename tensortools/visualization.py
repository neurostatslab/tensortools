"""
Plotting routines for CP decompositions
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_factors', 'plot_objective', 'plot_similarity']


def plot_objective(ensemble, partition='train', ax=None, jitter=0.1,
                   scatter_kw=dict(), line_kw=dict()):
    """Plots objective function as a function of model rank.

    Parameters
    ----------
    ensemble : Ensemble object
        holds optimization results across a range of model ranks
    partition : string, one of: {'train', 'test'}
        specifies whether to plot the objective function on the training
        data or the held-out test set.
    ax : matplotlib axis (optional)
        axis to plot on (defaults to current axis object)
    jitter : float (optional)
        amount of horizontal jitter added to scatterpoints (default=0.1)
    scatter_kw : dict (optional)
        keyword arguments for styling the scatterpoints
    line_kw : dict (optional)
        keyword arguments for styling the line
    """

    if ax is None:
        ax = plt.gca()

    if partition == 'train':
        pass
    elif partition == 'test':
        raise NotImplementedError('Cross-validation is on the TODO list.')
    else:
        raise ValueError("partition must be 'train' or 'test'.")

    # compile statistics for plotting
    x, obj, min_obj = [], [], []
    for rank in sorted(ensemble.results):
        # reconstruction errors for rank-r models
        o = ensemble.objectives(rank)
        obj.extend(o)
        x.extend(np.full(len(o), rank))
        min_obj.append(min(o))

    # add horizontal jitter
    ux = np.unique(x)
    x = np.array(x) + (np.random.rand(len(x))-0.5)*jitter

    # make plot
    ax.scatter(x, obj, **scatter_kw)
    ax.plot(ux, min_obj, **line_kw)
    ax.set_xlabel('model rank')
    ax.set_ylabel('objective')

    return ax


def plot_similarity(ensemble, ax=None, jitter=0.1,
                    scatter_kw=dict(), line_kw=dict()):
    """Plots similarity across optimization runs as a function of model rank.

    Parameters
    ----------
    ensemble : Ensemble object
        holds optimization results across a range of model ranks
    ax : matplotlib axis (optional)
        axis to plot on (defaults to current axis object)
    jitter : float (optional)
        amount of horizontal jitter added to scatterpoints (default=0.1)
    scatter_kw : dict (optional)
        keyword arguments for styling the scatterpoints
    line_kw : dict (optional)
        keyword arguments for styling the line

    References
    ----------
    Ulrike von Luxburg (2010). Clustering Stability: An Overview.
    Foundations and Trends in Machine Learning.
    https://arxiv.org/abs/1007.1075

    """

    if ax is None:
        ax = plt.gca()

    # compile statistics for plotting
    x, sim, mean_sim = [], [], []
    for rank in sorted(ensemble.results):
        # reconstruction errors for rank-r models
        s = ensemble.similarities(rank)[1:]
        sim.extend(s)
        x.extend(np.full(len(s), rank))
        mean_sim.append(np.mean(s))

    # add horizontal jitter
    ux = np.unique(x)
    x = np.array(x) + (np.random.rand(len(x))-0.5)*jitter

    # make plot
    ax.scatter(x, sim, **scatter_kw)
    ax.plot(ux, mean_sim, **line_kw)

    ax.set_xlabel('model rank')
    ax.set_ylabel('model similarity')
    ax.set_ylim([0, 1.1])

    return ax


def plot_factors(U, plots='line', fig=None, axes=None, scatter_kw=dict(),
                 line_kw=dict(), bar_kw=dict(), **kwargs):
    """Plots a KTensor.

    Note: Each keyword option is broadcast to all modes of the KTensor. For
    example, if `U` is a 3rd-order tensor (i.e. `U.ndim == 3`) then
    `plot_factors(U, plots=['line','bar','scatter'])` plots all factors for the
    first mode as a line plot, the second as a bar plot, and the third mode as
    a scatterplot. But, thanks to broadcasting semantics,
    `plot_factors(U, color='line')` produces line plots for each mode.

    Parameters
    ----------
    U : KTensor
        Kruskal tensor to be plotted.

    plots : str or list
        One of {'bar','line','scatter'} to specify the type of plot for each
        factor. The default is 'line'.
    fig : matplotlib Figure object
        If provided, add plots to the specified figure. The figure must have a
        sufficient number of axes objects.
    axes : 2d numpy array of matplotlib Axes objects
        If provided, add plots to the specified figure.
    scatter_kw : dict or sequence of dicts
        Keyword arguments provided to scatterplots. If a single dict is
        provided, these options are broadcasted to all modes.
    line_kw : dict or sequence of dicts
        Keyword arguments provided to line plots. If a single dict is provided,
        these options are broadcasted to all modes.
    bar_kw : dict or sequence of dicts
        Keyword arguments provided to bar plots. If a single dict is provided,
        these options are broadcasted to all modes.
    **kwargs : dict
        Additional keyword parameters are passed to the `subplots(...)`
        function to specify options such as `figsize` and `gridspec_kw`. See
        `matplotlib.pyplot.subplots(...)` documentation for more info.
    """

    # ~~~~~~~~~~~~~
    # PARSE OPTIONS
    # ~~~~~~~~~~~~~
    kwargs.setdefault('figsize', (8, U.rank))

    # parse optional inputs
    plots = _broadcast_arg(U, plots, str, 'plots')
    bar_kw = _broadcast_arg(U, bar_kw, dict, 'bar_kw')
    line_kw = _broadcast_arg(U, line_kw, dict, 'line_kw')
    scatter_kw = _broadcast_arg(U, scatter_kw, dict, 'scatter_kw')

    # default scatterplot options
    for sckw in scatter_kw:
        sckw.setdefault('edgecolor', 'none')
        sckw.setdefault('s', 10)

    # ~~~~~~~~~~~~~~
    # SETUP SUBPLOTS
    # ~~~~~~~~~~~~~~
    if fig is None and axes is None:
        fig, axes = plt.subplots(U.rank, U.ndim, **kwargs)
        # make sure axes is a 2d-array
        if U.rank == 1:
            axes = axes[None, :]

    # if axes are passed in, identify figure
    elif fig is None:
        fig = axes[0, 0].get_figure()

    # if figure is passed, identify axes
    else:
        axes = np.array(fig.get_axes(), dtype=object).reshape(U.rank, U.ndim)

    # main loop, plot each factor
    plot_obj = np.empty((U.rank, U.ndim), dtype=object)
    for r in range(U.rank):
        for i, f in enumerate(U):
            # start plots at 1 instead of zero
            x = np.arange(1, f.shape[0]+1)

            # determine type of plot
            if plots[i] == 'bar':
                plot_obj[r, i] = axes[r, i].bar(x, f[:, r], **bar_kw[i])
                axes[r, i].set_xlim(0, f.shape[0]+1)
            elif plots[i] == 'scatter':
                plot_obj[r, i] = axes[r, i].scatter(x, f[:, r], **scatter_kw[i])
                axes[r, i].set_xlim(0, f.shape[0])
            elif plots[i] == 'line':
                plot_obj[r, i] = axes[r, i].plot(f[:, r], '-', **line_kw[i])
                axes[r, i].set_xlim(0, f.shape[0])
            else:
                raise ValueError('invalid plot type')

            # format axes
            axes[r, i].locator_params(nbins=4)
            axes[r, i].spines['top'].set_visible(False)
            axes[r, i].spines['right'].set_visible(False)
            axes[r, i].xaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_tick_params(direction='out')
            axes[r, i].yaxis.set_ticks_position('left')
            axes[r, i].xaxis.set_ticks_position('bottom')

            # remove xticks on all but bottom row
            if r != U.rank-1:
                plt.setp(axes[r, i].get_xticklabels(), visible=False)

    # link y-axes within columns
    for i in range(U.ndim):
        yl = [a.get_ylim() for a in axes[:, i]]
        y0, y1 = min([y[0] for y in yl]), max([y[1] for y in yl])
        [a.set_ylim((y0, y1)) for a in axes[:, i]]

    # format y-ticks
    for r in range(U.rank):
        for i in range(U.ndim):
            # only two labels
            ymin, ymax = np.round(axes[r, i].get_ylim(), 2)
            axes[r, i].set_ylim((ymin, ymax))

            # remove decimals from labels
            if ymin.is_integer():
                ymin = int(ymin)
            if ymax.is_integer():
                ymax = int(ymax)

            # update plot
            axes[r, i].set_yticks([ymin, ymax])

    plt.tight_layout()

    return fig, axes, plot_obj


def _broadcast_arg(U, arg, argtype, name):
    """Broadcasts plotting option `arg` to all factors.

    Args:
        U : KTensor
        arg : argument provided by the user
        argtype : expected type for arg
        name : name of the variable, used for error handling

    Returns:
        iterable version of arg of length U.ndim
    """

    # if input is not iterable, broadcast it all dimensions of the tensor
    if arg is None or isinstance(arg, argtype):
        return [arg for _ in range(U.ndim)]

    # check if iterable input is valid
    elif np.iterable(arg):
        if len(arg) != U.ndim:
            raise ValueError('Parameter {} was specified as a sequence of '
                             'incorrect length. The length must match the '
                             'number of tensor dimensions '
                             '(U.ndim={})'.format(name, U.ndim))
        elif not all([isinstance(a, argtype) for a in arg]):
            raise TypeError('Parameter {} specified as a sequence of '
                            'incorrect type. '
                            'Expected {}.'.format(name, argtype))
        else:
            return arg

    # input is not iterable and is not the corrent type.
    else:
        raise TypeError('Parameter {} specified as a {}.'
                        ' Expected {}.'.format(name, type(arg), argtype))
