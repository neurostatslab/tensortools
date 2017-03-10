import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import numpy as np

def tensor_raster(data, axes=None, column=0, ncols=1, colors=None, title=None, background='dark', interpolation='none', **subplots_kw):
    """
    Generates spike rasters with each neuron stacked on top of one another

    Usage
    -----
    >>> X   # shape == (n_neurons, n_time_points, n_trials)
    >>> tensor_raster(X)

    Parameters
    ----------
        data: array of the form [n_neuron, n_time, n_trial]
        grid_spec: matplotlib.GridSpec (default: GridSpec(n_neuron, 1))
        colors: fixed colormap to use for all neurons (default: hand-picked by Alex)
        n_colors: number of colormap colors (default: 9)
        background: colormap specified by string (default: 'dark')
        interpolation: inteprolation argument to imshow (default: None)
    """
    if colors is None and background in ['white', 'light']:
        colors = [(0, 0, 0), (0, 0, 0.7), (0.7, 0, 0), (0, 0.8, 0), (0.7, 0, 0.7), (0.9, 0.4, 0)]
    elif colors is None and background  in ['black', 'dark']:
        colors = [(1, 1, 1), (0.3, 1, 0.3), (1, 0.1, 0.1), (1, 0.2, 1), (1, 1, 0), (0, 0.9, 1)]

    n_colors = len(colors)
    n_neuron = data.shape[0]

    # catch keywords intended for gridspec
    gridspec_kw = {}
    for k in list(subplots_kw.keys()):
        if k in ['hspace', 'wspace']:
            gridspec_kw[k] = subplots_kw[k]
            subplots_kw.pop(k, None)

    # setup axes
    if axes is None:
        fig, axes = plt.subplots(nrows=n_neuron, ncols=ncols, gridspec_kw=gridspec_kw, **subplots_kw)
    else:
        fig = None

    axcolumn = axes if axes.ndim == 1 else axes[:, column]
    images = []

    for i, neuron, ax in zip(range(n_neuron), data, axcolumn):
        
        # set up colormap
        if background in ['black', 'dark']:
            cm = _dark_colormap(colors[i % n_colors])
        elif background in ['white', 'light']:
            cm = _light_colormap(colors[i % n_colors])
        else:
            raise ValueError('Background argument misspecified.')
        
        # plot neuron raster
        img = ax.imshow(np.nan_to_num(data[i].T), cmap=cm, interpolation=interpolation, aspect='auto')
        images.append(img)

        ax.axis('tight')
        ax.axis('off')

    if title is not None:
        axcolumn[0].set_title(title)

    return fig, axes, images

def _light_colormap(c):
    r,g,b = colorConverter.to_rgb(c)
    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (1.0,   r,   r)),
             'green': ((0.0, 1.0, 1.0),
                       (1.0,   g,   g)),
             'blue':  ((0.0, 1.0, 1.0),
                       (1.0,   b,   b))
            }
    return LinearSegmentedColormap('_', cdict)

def _dark_colormap(c):
    r,g,b = colorConverter.to_rgb(c)
    cdict = {'red':   ((0.0, 0.0, 0.0),
                       (1.0,   r,   r)),
             'green': ((0.0, 0.0, 0.0),
                       (1.0,   g,   g)),
             'blue':  ((0.0, 0.0, 0.0),
                       (1.0,   b,   b))
            }
    return LinearSegmentedColormap('_', cdict)
