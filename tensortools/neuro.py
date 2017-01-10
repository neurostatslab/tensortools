"""
Functions tailored to multi-trial neural data.
"""

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from tensortools.kruskal import _validate_kruskal
from tensorly import kruskal_to_tensor   
from jetpack import breathe, bars

DEFAULT_COLOR_CYCLE = [(0.4470588235294118, 0.6196078431372549, 0.807843137254902),
                       (1.0, 0.6196078431372549, 0.2901960784313726),
                       (0.403921568627451, 0.7490196078431373, 0.3607843137254902),
                       (0.9294117647058824, 0.4, 0.36470588235294116),
                       (0.6784313725490196, 0.5450980392156862, 0.788235294117647),
                       (0.6588235294117647, 0.47058823529411764, 0.43137254901960786),
                       (0.9294117647058824, 0.592156862745098, 0.792156862745098),
                       (0.6352941176470588, 0.6352941176470588, 0.6352941176470588),
                       (0.803921568627451, 0.8, 0.36470588235294116),
                       (0.42745098039215684, 0.8, 0.8549019607843137)]

def interact_reconstruction(data, model, avg_trial=False, condition=None, color_cycle=DEFAULT_COLOR_CYCLE, **kwargs):
    """Returns a function for interacive plotting of model reconstruction error
    """

    ndim, rank = _validate_kruskal(model)

    if ndim != 3 or data.ndim != 3:
        raise ValueError('Data and model must be a 3rd-order tensor.')

    N, T, K = data.shape
    tax = np.linspace(0,2,T)

    if avg_trial:
        Xest = kruskal_to_tensor(model)
        if condition is None:
            condition = np.ones(K, np.int)
        Cavg = [np.mean(model[2][condition == c], axis=0) for c in np.unique(condition)]
        Xavg = [np.mean(data[:, :, condition == c], axis=2) for c in np.unique(condition)]
        Xavg_est = [np.mean(Xest[:, :, condition == c], axis=2) for c in np.unique(condition)]
        neuron_factor_lim = (np.min(model[0]), np.max(model[0]))
        trial_factor_lim = (min([np.min(c) for c in Cavg]), max([np.max(c) for c in Cavg]))

    # plotting function with trial averaging
    def _yes_trial_avg(neuron):
        fig, axes = plt.subplots(2, 2, gridspec_kw=dict(width_ratios=[3,1]), **kwargs)

        [axes[0,0].plot(tax, x[neuron], '-', color=c, lw=3, alpha=0.3) for x, c in zip(Xavg, color_cycle)]
        [axes[0,0].plot(tax, x[neuron], '--', color=c, lw=3) for x, c in zip(Xavg_est, color_cycle)]
        axes[0,0].set_ylim((0, max([np.max(x) for x in Xavg])))

        [axes[1,0].plot(tax, model[0][neuron]*model[1]*cavg, '-', lw=3, color=c, alpha=0.5) for cavg, c in zip(Cavg, color_cycle)]
        axes[1,0].set_title('latent factors', y=1)
        axes[1,0].set_xlabel('time (s)')
        
        axes[0,1].bar(range(1,rank+1), model[0][neuron], align='center', color=(0.5,0.5,0.5))
        axes[0,1].axhline(0, color='k', lw=1)
        axes[0,1].set_xlim([0, rank+1])
        axes[0,1].set_xticks(range(1,rank+1))
        axes[0,1].set_title('neuron loadings')
        axes[0,1].set_ylim(neuron_factor_lim)

        bars(np.array(Cavg).T, ax=axes[1,1], colors=color_cycle)
        axes[1,1].set_title('trial loadings')
        axes[1,1].axhline(0, color='k', lw=1)
        axes[1,1].set_ylim(trial_factor_lim)

        plt.tight_layout()
        [breathe(ax=ax) for ax in axes.ravel()]

    # plotting function without trial averaging
    def _no_trial_avg(neuron, trial):
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

    if avg_trial:
        return interact(_yes_trial_avg, neuron=(0, N-1))
    else:
        return interact(_no_trial_avg, neuron=(0, N-1), trial=(0, K-1))
