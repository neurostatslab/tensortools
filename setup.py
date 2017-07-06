try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import tensortools
version = tensortools.__version__

config = {
    'name': 'tensortools',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tools for Tensor Decomposition.',
    'author': 'Alex Williams',
    'author_email': 'alex.h.willia@gmail.com',
    'version': version,
    'url': 'https://github.com/ahwillia/tensortools',
    'license': 'MIT'
}

setup(**config)
