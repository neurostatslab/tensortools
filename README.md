## What is TensorTools?

TensorTools is a bare bones Python package for fitting and visualizing [canonical polyadic (CP) tensor decompositions](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) of higher-order data arrays.
Check out the [`examples/`](./examples) folder for usage.

#### Fitting an ensemble of models

CP decompositions can be caught in local minima during optimization **[2]**.
If our data is held in `data_tensor` (a multi-dimensional numpy array), we can fit an ensemble of models with:

```python
from tensortools import fit_ensemble

# Fit models with 1-10 components. For each model rank, fit from 3 random initializations. 
results = fit_ensemble(data_tensor, range(1,11), replicates=3)
```

#### Diagnostic Plots

TensorTools provides code to visualize fitted ensembles of CP decomposition models. These plots help assess (a) whether models are becoming stuck in highly suboptimal local minima, (b) whether the final model parameters are sensitive to the initialization. Together these can help a practitioner determine the appropriate number of latent components in the model.

```python
from tensortools import plot_error, plot_similarity

plot_error(results)
plot_similarity(results)
```

#### Visualizing Factors

After fitting your models and assessing their goodness of fit (see above), TensorTools provides methods to visualize the low-rank factors.

```python
from tensortools import plot_factors

# plot the best model with R = 5 components
R = 5
plot_factors(results[R]['factors'][0])
```

## What *isn't* TensorTools?

TensorTools does not support many other interesting tensor models, such as Tucker decompositions and tensor regression problems.
See **[1]** for a broad review of many tensor models.
Additionally, TensorTools does not support specialized data structures for sparse tensors (see **[3]**).
Some of these models are supported by other tensor packages:

* MATLAB
    * [TensorToolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html)
    * [TensorLab](http://www.tensorlab.net/)
* Python
    * [TensorLy](https://tensorly.github.io/stable/index.html)

## Installation

For now, install manually from the command line:

```bash
$ git clone https://github.com/ahwillia/tensortools
$ cd tensortools
$ python setup.py install
```

Or, if you plan to modify the source code yourself you can use:

```bash
$ python setup.py develop
```

which will conveniently incorporate any edits you make to the code after you restart the Python kernel. Open [an issue](https://github.com/ahwillia/tensortools/issues) if you have trouble installing tensortools.

## References

[**[ 1 ]**](http://www.sandia.gov/~tgkolda/pubs/pubfiles/TensorReview.pdf) Kolda TG, Bader BW (2009). "Tensor Decompositions and Applications." *SIAM Review* 51:3, 455–500.

[**[ 2 ]**](http://dx.doi.org/10.1002/cem.1236) Comon P, Luciani X, de Almeida ALF (2009). "Tensor decompositions, alternating least squares and other tales." *Journal of Chemometrics* 23:7-8, 393–405.

[**[ 3 ]**](http://www.sandia.gov/~tgkolda/pubs/pubfiles/SAND2006-7592.pdf) Bader BW, Kolda TG (2006). "Efficient MATLAB computations with sparse and factored tensors." *Technical Report, Sandia National Laboratories.*
