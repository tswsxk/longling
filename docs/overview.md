![longling logo](_static/longling_logo.png)

# longling

[![Documentation Status](https://readthedocs.org/projects/longling/badge/?version=latest)](https://longling.readthedocs.io/zh/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/longling.svg)](https://pypi.python.org/pypi/longling)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/longling.svg)](https://pypi.python.org/pypi/longling)
[![Build Status](https://www.travis-ci.org/tswsxk/longling.svg?branch=master)](https://www.travis-ci.org/tswsxk/longling)
[![codecov](https://codecov.io/gh/tswsxk/longling/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/longling)
[![Download](https://img.shields.io/pypi/dm/longling.svg?style=flat)](https://pypi.python.org/pypi/longling)
[![License](https://img.shields.io/github/license/tswsxk/longling)](LICENSE)
![CodeSize](https://img.shields.io/github/languages/code-size/tswsxk/longling)
![CodeLine](https://img.shields.io/tokei/lines/github/tswsxk/longling)

## Overview
The project contains several modules for different purposes:

* [lib](submodule/lib/index.html) serves as the basic toolkit that can be used in any place without extra dependencies.

* [ML](submodule/ML/index.html) provides many interfaces to quickly build machine learning tools.

## Quick scripts
The project provide several cli scripts to help construct different 
architecture.

### Neural Network
* [glue](submodule/ML/MxnetHelper/glue.html) for mxnet-gluon



### CLI
Provide several general tools, consistently invoked by: 

```shell
longling $subcommand $parameters1 $parameters2
```

To see the `help` information:
```shell
longling -- --help
longling $subcommand --help
```

Take a glance on [all available cli](submodule/cli.html).

The cli tools is constructed based on [fire](https://github.com/google/python-fire). 
Refer to the [documentation](https://github.com/google/python-fire/blob/master/docs/using-cli.md) for detailed usage.

#### Demo

##### Split dataset

target: split a dataset into `train/valid/test`

```shell
longling train_valid_test $filename1 $filename2 -- --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 
```

Similar commands:

* `train_test`

```shell
longling train_test $filename1 -- --train_ratio 0.8 --test_ratio 0.2 
```

* `train_valid`

```shell
longling train_valid $filename1 -- --train_ratio 0.8 --valid_ratio 0.2 
```

* Cross Validation `kfold`

```shell
longling kfold $filename1 $filename2 -- --n_splits 5
```

##### Display the tree of content

```shell
longling toc .
```

such as 
```text
/
├── __init__.py
├── __pycache__/
│   ├── __init__.cpython-36.pyc
│   └── toc.cpython-36.pyc
└── toc.py
```

##### Quickly construct a project

```shell
longling arch 
```

or you can also directly copy the template files

```shell
longling arch-cli
```
To be noticed that, you need to check `$VARIABLE` in the template files.

##### Result Analysis
The cli tools for result analysis is specially designed for json result format:

```shell
longling  max $filename $key1 $key2 $key3
longling  amax $key1 $key2 $key3 --src $filename
```

For the composite key like `{'prf':{'avg': {'f1': 0.77}}}`, the key should be presented as `prf:avg:f1`.
Thus, all the key used in the result file should not contain `:`.