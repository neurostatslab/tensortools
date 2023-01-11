from setuptools import setup, find_packages


NAME = 'tensortools'
DESCRIPTION = 'Tools for Tensor Decomposition.'
AUTHOR = 'Alex Williams and N. Benjamin Erichson'
EMAIL = 'alex.h.willia@gmail.com'
VERSION = "0.4"
URL = 'https://github.com/ahwillia/tensortools'
LICENSE = 'MIT'

install_requires = [
    'numpy',
    'scipy',
    'tqdm',
    'munkres',
    'numba',
    'matplotlib',
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    install_requires=install_requires,
    python_requires='>=3',
    project_urls={
        'Documentation': 'https://tensortools-docs.readthedocs.io/',
    },

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='tensor decomposition, canonical decomposition, parallel factors',
    packages=find_packages(exclude=['tests*']),
)
