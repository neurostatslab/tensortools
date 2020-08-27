Tensortools
-----------
[![][docs-stable-img]][docs-stable-url]  [![][license-img]][license-url]

TensorTools is a bare bones Python package for fitting and visualizing [canonical polyadic (CP) tensor decompositions](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) of higher-order data arrays. Check out the [documentation][docs-stable-url] and [`examples/`](./examples) folder for more detailed information.


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://tensortools-docs.readthedocs.io/en/latest/


[license-img]: https://img.shields.io/github/license/mashape/apistatus.svg
[license-url]: https://github.com/ahwillia/tensortools/blob/master/LICENSE.md


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
