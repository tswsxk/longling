# longling

[![Documentation Status](https://readthedocs.org/projects/longling/badge/?version=latest)](https://longling.readthedocs.io/zh/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/longling.svg)](https://pypi.python.org/pypi/longling)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/longling.svg)](https://pypi.python.org/pypi/longling)

This project aims to provide some handy toolkit functions to help construct the
architecture. 
Full documentation [here](https://longling.readthedocs.io/zh/latest/index.html).

## Installation

### pip
```bash
pip install longling
```
### source
clone the repository and then run `python setup.py install`:
```bash
git clone https://github.com/tswsxk/longling.git
python setup.py install
```

### Notation
Due to the possible multi version of deep learning frameworks like 
mxnet(for cpu) and mxnet-cu90(for gpu, with cuda version 9.0), 
it is good to install such frameworks in advance. 
For swift installation, use `--no-dependencies` option as follows:
```bash
# pip
pip install longling --no-dependencies
# source
python setup.py install --no-dependencies
```

## Overview
The project contains several modules for different purposes:

* [lib](https://longling.readthedocs.io/zh/latest/submodule/lib/index.html) serves as the basic toolkit that can be used in any place without extra dependencies.

* [ML](https://longling.readthedocs.io/zh/latest/submodule/ML/index.html) provides many interfaces to quickly build machine learning tools.

## Quick scripts
The project provide several cli scripts to help construct different 
architecture.

### Neural Network
* [glue](https://longling.readthedocs.io/zh/latest/submodule/ML/MxnetHelper/glue.html) for mxnet-gluon


### CLI
