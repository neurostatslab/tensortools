"""
Functions tailored to multi-trial neural data.
"""

from .kruskal import _validate_kruskal
from ipywidgets import interact

def interact_reconstruction(data, model):
    """Returns a function for interacive plotting of model reconstruction error
    """

    ndim, rank = _validate_kruskal(model)

    if ndim != 3 or data.ndim != 3:
        raise ValueError('Data and model must be a 3rd-order tensor.')

    N, T, K = data.shape

    def _f(n, k, legend_on):
        plt.plot(data[n, :, k], '-k', lw=3, label='data')
        plt.plot(model[0][n]*model[1]*model[2][k], '-k', lw=3, 'model')
        if legend_on:
            plt.legend(loc='best')

    return interact(_f, neuron=(0, N-1), trial=(0, K-1), legend=(True, False))
