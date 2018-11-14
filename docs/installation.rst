.. -*- rst -*-

Getting Started
===============

Installation
------------
The latest version of ``tensortools`` can be found on `GitHub <https://github.com/ahwillia/tensortools>`_. You can also install tensortools from the command line:

.. code-block:: bash
    
    pip install git+https://github.com/ahwillia/tensortools


Quick Example
-------------

Here is a short example script you may try running to check your installation. This code generates a ``25 x 25 x 25`` tensor with a low-rank component plus noise. We then use ``tensortools`` to fit a CP decomposition model to recover the low-rank factors. Since fitting a low-rank decomposition is a nonconvex problem, we fit the model twice from different starting parameters and show that we nevertheless recover the same result (up to a permutation and sign-flipping of the low-rank factors).

.. code-block:: python
    
    import tensortools as tt
    import numpy as np
    import matplotlib.pyplot as plt

    # Make synthetic dataset.
    I, J, K, R = 25, 25, 25, 4  # dimensions and rank
    X = tt.randn_ktensor((I, J, K), rank=R).full()
    X += np.random.randn(I, J, K)  # add noise

    # Fit CP tensor decomposition (two times).
    U = tt.cp_als(X, rank=R, verbose=True)
    V = tt.cp_als(X, rank=R, verbose=True)

    # Compare the low-dimensional factors from the two fits.
    fig, _, _ = tt.plot_factors(U.factors)
    tt.plot_factors(V.factors, fig=fig)

    # Align the two fits and print a similarity score.
    sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
    print(sim)

    # Plot the results again to see alignment.
    fig, ax, po = tt.plot_factors(U.factors)
    tt.plot_factors(V.factors, fig=fig)

    # Show plots.
    plt.show()
