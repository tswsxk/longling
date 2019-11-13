# longling

[![Documentation Status](https://readthedocs.org/projects/longling/badge/?version=latest)](https://longling.readthedocs.io/zh/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/longling.svg)](https://pypi.python.org/pypi/longling)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/longling.svg)](https://pypi.python.org/pypi/longling)

This project aims to provide some handy toolkit functions to help construct the
architecture. 
Full documentation [here](https://longling.readthedocs.io/zh/latest/index.html).

## Installation

### pip
```shell script
pip install longling
```
### source
clone the repository and then run `python setup.py install`:
```shell script
git clone https://github.com/tswsxk/longling.git
python setup.py install
```

### Notation
Due to the possible multi version of deep learning frameworks like 
mxnet(for cpu) and mxnet-cu90(for gpu, with cuda version 9.0), 
it is good to install such frameworks in advance. 
For swift installation, use `--no-dependencies` option as follows:
```shell script
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
All available cli tools are listed as follows:

#### longling cli
Provide several general tools, consistently invoked by:
```shell script
longling $subcommand $parameters1 $parameters2
```
To see the `help` information:
```shell script
longling -- --help
longling $subcommand --help
```
##### Subcommand demo

###### Dataset
Split the dataset into `train/valid/test`:
```shell script
longling train_valid_test $filename1 $filename2 -- --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 
```
Similar commands:
* `train_test`
```shell script
longling train_test $filename1 -- --train_ratio 0.8 --test_ratio 0.2 
```
* `train_valid`
```shell script
longling train_valid $filename1 -- --train_ratio 0.8 --valid_ratio 0.2 
```
* Cross Validation `kfold`
```shell script
longling kfold $filename1 $filename2 -- --n_splits 5
```

###### Display the tree of content
```shell script
longling toc .
```

##### Quickly construct a project
```shell script
longling arch 
```

##### Result Analysis
The cli tools for result analysis is specially designed for json result format:
```shell script
longling  max
```
