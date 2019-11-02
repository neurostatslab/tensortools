from matplotlib.colors import LinearSegmentedColormap, colorConverter
import matplotlib.pyplot as plt
import numpy as np


def sort_rsq(X, Xest):
    Xbar = np.mean(X, axis=(0, 1), keepdims=True)
    num = np.sum((Xest - X)**2, axis=(0, 1))
    denom = np.sum((X - Xbar)**2, axis=(0, 1))
    return np.argsort(num / denom)


def attribution_heatmap(model, data, n_columns=3):
    pred = []
    for i in range(model.rank):
        idx = set(range(model.rank)) - {i}
        pred.append(model.predict(skip_dims=list(idx)))

    map_ids = np.argmax(pred, axis=0)
    pred = np.sum(pred, axis=0)
    ordering = sort_rsq(data, pred)

    n_rows = data.shape[-1] // n_columns
    if (n_rows * n_columns) < data.shape[-1]:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows, 2 * n_columns, figsize=(n_columns * 4, n_rows * 2),
        sharey=True, sharex=True)

    for n, ax in zip(ordering, axes[:, ::2].ravel()):
        ax.imshow(data[:, :, n], cmap="gray_r", aspect="auto")

    for n, ax in zip(ordering, axes[:, 1::2].ravel()):
        multi_imshow(ax, pred[:, :, n], map_ids[:, :, n])

    return fig, axes


def pred_heatmap(model, data, n_columns=3):

    pred = model.predict()
    ordering = sort_rsq(data, pred)

    n_rows = data.shape[-1] // n_columns
    if (n_rows * n_columns) < data.shape[-1]:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows, 2 * n_columns, figsize=(n_columns * 4, n_rows * 2),
        sharey=True, sharex=True)

    for n, ax in zip(ordering, axes[:, ::2].ravel()):
        ax.imshow(
            data[:, :, n], cmap="gray_r", aspect="auto",
            interpolation="none")

    for n, ax in zip(ordering, axes[:, 1::2].ravel()):
        ax.imshow(
            pred[:, :, n], cmap="Blues", aspect="auto",
            interpolation="none")

    return fig, axes


def parts_heatmap(model, data, start=0, stop=20):

    pred = []
    for i in range(model.rank):
        idx = set(range(model.rank)) - {i}
        pred.append(model.predict(skip_dims=list(idx)))

    full_pred = np.sum(pred, axis=0)
    ordering = sort_rsq(data, full_pred)[start:stop]

    n_rows = len(ordering)

    fig, axes = plt.subplots(
        n_rows, model.rank + 2, figsize=(model.rank * 3 + 3, n_rows * 3),
        sharey=True, sharex=True)

    for n, ax in zip(ordering, axes[:, 0]):
        ax.imshow(
            data[:, :, n], cmap="gray_r", aspect="auto",
            interpolation="none", clim=(0, 1))

    for n, ax in zip(ordering, axes[:, 1]):
        ax.imshow(
            full_pred[:, :, n], cmap="gray_r", aspect="auto",
            interpolation="none", clim=(0, 1))

    for k in range(model.rank):
        for n, ax in zip(ordering, axes[:, k + 2]):
            ax.imshow(
                pred[k][:, :, n], cmap="Blues", aspect="auto",
                interpolation="none", clim=(0, 1))

    return fig, axes


def residual_heatmap(model, data, n_columns=3):

    n_rows = data.shape[-1] // n_columns
    if (n_rows * n_columns) < data.shape[-1]:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows, 2 * n_columns, figsize=(n_columns * 4, n_rows * 2),
        sharey=True, sharex=True)

    pred = model.predict()
    resids = data - pred
    ordering = sort_rsq(data, pred)

    for n, ax in zip(ordering, axes[:, ::2].ravel()):
        ax.imshow(data[:, :, n], cmap="gray_r", aspect="auto")

    for n, ax in zip(ordering, axes[:, 1::2].ravel()):
        cmax = max(abs(np.percentile(resids[:, :, n], (10, 90))))
        ax.imshow(
            resids[:, :, n],
            cmap="bwr",
            aspect="auto",
            clim=(-cmax, cmax)
        )

    return fig, axes


def multi_imshow(ax, img_data, map_ids, cmaps=None):
    """
    Creates an image with multiple color maps.

    Parameters
    ----------
    ax : matplotlib Axis object
    data : ndarray, 2-dimensional image data.
    map_ids : ndarray, same shape as data.
    cmaps : list of strings, specifying colormaps.
    """

    if cmaps is None:
        cmaps = [
            simple_cmap("w", "#0000E8"),
            simple_cmap("w", "#FF8B00"),
            simple_cmap("w", "#009E05"),
            simple_cmap("w", "#DE0300"),
            simple_cmap("w", "#9400DE"),
            simple_cmap("w", "#8F4100"),
            simple_cmap("w", "#F50085"),
        ]

    for i, cm in enumerate(cmaps):

        mask = (map_ids != i)
        if np.all(mask):
            continue

        cm.set_bad(alpha=0.0)
        md = np.ma.array(img_data, mask=mask)
        ax.imshow(md, interpolation="none", aspect="auto", cmap=cm)


def simple_cmap(*colors, name='none'):
    """Create a colormap from a sequence of rgb values.

    cmap = simple_cmap((1,1,1), (1,0,0)) # white to red colormap
    cmap = simple_cmap('w', 'r')         # white to red colormap
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # make sure colors are specified as rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})
