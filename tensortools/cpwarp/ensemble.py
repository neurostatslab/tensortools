import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt


def shifted_ensemble(
        X, ranks, shifts, n_replicates=15, **fit_kw):

    results = [[[] for s in shifts] for r in ranks]

    fit_kw["verbose"] = False
    pbar = tqdm(total=(len(ranks) * len(shifts) * n_replicates))

    for i, r in enumerate(ranks):
        for j, s in enumerate(shifts):
            for _ in range(n_replicates):
                results[i][j].append(
                    shifted_ncp3_hals(X, r, max_shift=s, **fit_kw))
                pbar.update(1)

    pbar.close()
    return Ensemble(results, ranks, shifts, n_replicates)


class ShiftEnsemble:

    def __init__(self, raw_results, ranks, shifts, n_replicates):

        self.ranks = ranks
        self.shifts = shifts
        self.n_replicates = n_replicates
        self.raw_results = raw_results

        I, J, K = len(ranks), len(shifts), n_replicates

        self.losses = np.empty((I, J, K))
        for i, j, k in itertools.product(range(I), range(J), range(K)):
            self.losses[i, j, k] = raw_results[i][j][k][1][-1]

    def select_model(self, i, j, k=0):
        _k = np.argsort(self.losses[i, j])[k]
        return self.raw_results[i][j][_k][0]

    def plot_losses(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.shifts, np.min(self.losses, axis=-1).T, '.-')
        ax.set_ylabel("loss")
        ax.set_xlabel("max shift")
        ax.legend(self.ranks, title="# components", bbox_to_anchor=(1, .5))
