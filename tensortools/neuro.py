"""
Functions tailored to multi-trial neural data.
"""

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from tensortools.kruskal import _validate_kruskal
from jetpack import breathe

def interact_reconstruction(data, model, **kwargs):
    """Returns a function for interacive plotting of model reconstruction error
    """

    ndim, rank = _validate_kruskal(model)

    if ndim != 3 or data.ndim != 3:
        raise ValueError('Data and model must be a 3rd-order tensor.')

    N, T, K = data.shape
    tax = np.linspace(0,2,T)

    def _f(neuron, trial):
        fig, axes = plt.subplots(2, 2, gridspec_kw=dict(width_ratios=[3,1]), **kwargs)

        axes[0,0].plot(tax, data[neuron, :, trial], '-k', lw=3, label='data')
        est = np.sum(model[0][neuron]*model[1]*model[2][trial], axis=1)
        axes[0,0].plot(tax, est, '--r', lw=3, label='model')
        axes[0,0].set_ylim([0, np.max(data[neuron])+1e-3])
        axes[0,0].set_title('data (black) and reconstruction (red)')
        axes[0,0].set_xlabel('time (s)')

        axes[1,0].plot(tax, model[0][neuron]*model[1]*model[2][trial], '-b', lw=3, alpha=0.6)
        axes[1,0].set_ylim(axes[0,0].get_ylim())
        axes[1,0].set_title('latent factors', y=0.8)
        axes[1,0].set_xlabel('time (s)')
        
        axes[1,0].plot(tax, model[0][neuron]*model[1]*model[2][trial], '-b', lw=3, alpha=0.6)
        axes[1,0].set_ylim(axes[0,0].get_ylim())
        axes[1,0].set_title('latent factors', y=0.8)
        axes[1,0].set_xlabel('time (s)')
        

        loadings = ('neuron loadings', 'trial loadings')
        for ax, factors, t in zip(axes[:,1], (model[0][neuron], model[2][trial]), loadings):
            ax.bar(range(1,rank+1), factors, align='center', color=(0.5,0.5,0.5))
            ax.axhline(0, color='k', lw=1)
            ax.set_xlim([0,rank+1])
            ax.set_xticks(range(1,rank+1))
            ax.set_title(t)
        axes[0,1].set_ylim((np.min(model[0]), np.max(model[0])))
        axes[1,1].set_ylim((np.min(model[2]), np.max(model[2])))

        plt.tight_layout()
        [breathe(ax=ax) for ax in axes.ravel()]

    return interact(_f, neuron=(0, N-1), trial=(0, K-1))
