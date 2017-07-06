import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import numpy as np

def tensor_raster(*tensors, colors=None, share_cax=True, background='white', **subplots_kw):
    """
    Generates spike rasters with each neuron stacked on top of one another

    Usage
    -----
    >>> X, Y  # data tensors (n_neurons x n_timepoints x n_trials)
    >>> fig, axes, images = tensor_raster(X)
    >>> fig, axes, images = tensor_raster(X, Y) # plots tensors side by side

    Keyword Arguments
    -----------------
    colors, list: colors to use for each raster (default: hand-picked)
    share_cax, bool: If True, set the heatmap limits to be equal (default: True)
    background, str: whether background of raster heatmap is black or white (default: 'white')
    **subplots_kw: All additional keywords are passed to plt.subplots(...) function
    """
    if colors is None and background in ['white', 'w']:
        colors = [(0, 0, 0), (0, 0, 0.7), (0.7, 0, 0), (0, 0.8, 0), (0.7, 0, 0.7), (0.9, 0.4, 0)]
    elif colors is None and background  in ['black', 'k']:
        colors = [(1, 1, 1), (0.3, 1, 0.3), (1, 0.1, 0.1), (1, 0.2, 1), (1, 1, 0), (0, 0.9, 1)]

    n_colors = len(colors)
    n_neuron = tensors[0].shape[0]
    n_tensors = len(tensors)

    # setup axes
    fig, axes = plt.subplots(nrows=n_neuron, ncols=n_tensors, gridspec_kw=gridspec_kw, **subplots_kw)
    images = np.empty((n_neuron, n_tensors), dtype=object)

    for i in range(n_neuron):

        # set up colormap
        if background in ['k', 'black']:
            cm = _dark_colormap(colors[i % n_colors])
        elif background in ['w', 'white']:
            cm = _light_colormap(colors[i % n_colors])
        else:
            raise ValueError('Background argument misspecified.')

        # plot this neuron for all tensors
        for j in range(n_tensors):
            images[i, j] = axes[i, j].imshow(np.nan_to_num(tensors[j][i].T), cmap=cm, interpolation='none', aspect='auto')
            axes[i, j].axis('tight')
            axes[i, j].axis('off')

        if share_cax:
            old_clim = np.array([im.get_clim() for im in images[i,:]])
            new_clim = (np.min(old_clim), np.max(old_clim))
            [im.set_clim(new_clim) for im in images[i,:]]

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
