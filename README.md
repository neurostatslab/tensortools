Tensortools
-----------
[![][license-img]][license-url]

[license-img]: https://img.shields.io/github/license/mashape/apistatus.svg
[license-url]: https://github.com/ahwillia/tensortools/blob/master/LICENSE.md


TensorTools is a bare bones Python package for fitting and visualizing [canonical polyadic (CP) tensor decompositions](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) of higher-order data arrays. I originally developed this library for applications in neuroscience ([Williams et al., 2018](https://doi.org/10.1016/j.neuron.2018.05.015)), but the code could be helpful in other domains.

Installation
------------

From the command line run:

```
pip install git+https://github.com/ahwillia/tensortools
```

(You will need to have `git` installed for this command to work.)

Alternatively you can download the source code and install locally by running:

```
git clone https://github.com/ahwillia/tensortools
cd tensortools
pip install -e .
```

Quick Start
------------

Here's how to perform a parameter sweep over 1 - 9 components, and plot the reconstruction error and similarity diagnostics as a function of the model rank (these diagnostics are described in [Williams et al., 2018](https://doi.org/10.1016/j.neuron.2018.05.015)). The snippet also uses `plot_factors(...)` to plot the factors extracted by one of the models in the ensemble.

The method `"ncp_hals"` fits a nonnegative tensor decomposition, other methods are `"ncp_bcd"` (also nonnegative) and `"cp_als"` (unconstrained decomposition). See the [`tensortools/optimize/`](/tensortools/optimize) folder for the implementation of these algorithms.


```python
import tensortools as tt

data = # ... specify a numpy array holding the tensor you wish to fit

# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
ensemble = tt.Ensemble(fit_method="ncp_hals")
ensemble.fit(data, ranks=range(1, 9), replicates=4)

fig, axes = plt.subplots(1, 2)
tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
fig.tight_layout()

# Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
num_components = 2
replicate = 0
tt.plot_factors(ensemble.factors(num_components)[replicate])  # plot the low-d factors

plt.show()
```

Check out the scripts in the [`examples/`](/examples) folder for other short demos.

Time-Shifted Tensor Decompositions
----------------------------------

This repo contains a module `tensortools.cpwarp` which allows for *time-shifted tensor decompositions* of 3d-arrays. The motivation behind this model and some of its implementaional details are laid out in the following set of notes.

>  Alex H. Williams (2020). [Combining tensor decomposition and time warping models for multi-neuronal spike train analysis](https://doi.org/10.1101/2020.03.02.974014). *bioRxiv*. 2020.03.02.974014

A very similar model was previously proposed by [Mørup et al. (2008)](https://doi.org/10.1016/j.neuroimage.2008.05.062). Also see [Sorokin et al. (2020)](https://doi.org/10.1101/2020.03.04.976688) for an application of this model to neural data.

To fit this model, check out the script in [`examples/shift_cpd.py`](./examples/shift_cpd.py), which should reproduce Figure 4 from the [Williams (2020)](https://doi.org/10.1101/2020.03.02.974014) paper. I hope to upload a more detailed tutorial soon; until then, please refer to the papers cited above and reach out to me by email if you are interested in further details.

Citation
--------

If you found this resource useful, please consider citing [this paper](https://doi.org/10.1016/j.neuron.2018.05.015).

```
@ARTICLE{Williams2018,
  title    = "Unsupervised Discovery of Demixed, {Low-Dimensional} Neural
              Dynamics across Multiple Timescales through Tensor Component
              Analysis",
  author   = "Williams, Alex H and Kim, Tony Hyun and Wang, Forea and Vyas,
              Saurabh and Ryu, Stephen I and Shenoy, Krishna V and Schnitzer,
              Mark and Kolda, Tamara G and Ganguli, Surya",
  journal  = "Neuron",
  volume   =  98,
  number   =  6,
  pages    = "1099--1115.e8",
  month    =  jun,
  year     =  2018,
}
```
