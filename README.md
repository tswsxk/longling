![longling logo](docs/_static/longling_logo.png)

# longling

[![VERSION](https://img.shields.io/pypi/pyversions/longling)](https://pypi.python.org/pypi/longling)
[![PyPI](https://img.shields.io/pypi/v/longling.svg)](https://pypi.python.org/pypi/longling)
[![test](https://github.com/tswsxk/longling/actions/workflows/python-test.yml/badge.svg?branch=master)](https://github.com/tswsxk/longling/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/tswsxk/longling/branch/main/graph/badge.svg?token=MQ11mcq0p6)](https://codecov.io/gh/tswsxk/longling)
[![Documentation Status](https://readthedocs.org/projects/longling/badge/?version=latest)](https://longling.readthedocs.io/zh/latest/index.html)
[![Download](https://img.shields.io/pypi/dm/longling.svg?style=flat)](https://pypi.python.org/pypi/longling)
[![License](https://img.shields.io/github/license/tswsxk/longling)](LICENSE)
![CodeSize](https://img.shields.io/github/languages/code-size/tswsxk/longling)
![CodeLine](https://img.shields.io/tokei/lines/github/tswsxk/longling)

This project aims to provide some handy toolkit functions to help construct the
architecture. 
Full documentation [here](https://longling.readthedocs.io/zh/latest/index.html).

## Installation

### pip

```shell
pip install longling
```

### source
clone the repository and then run `pip install .`:

```shell
git clone https://github.com/tswsxk/longling.git
cd longling
pip install .
```

### Notation
Due to the possible multi version of deep learning frameworks like 
mxnet(for cpu) and mxnet-cu90(for gpu, with cuda version 9.0), 
it is good to install such frameworks in advance. 
For swift installation, use `--no-dependencies` option as follows:

```shell
# pip
pip install longling --no-dependencies
# source
python setup.py install --no-dependencies
```

### Help for windows user

Due to the potential compile error in windows, some required package may not be installed as expected.
To deal with this issue, pre-compiled binaries are advised.
You can go to [lfd.uci.edu/~gohlke/pythonlibs](https://www.lfd.uci.edu/~gohlke/pythonlibs) 
and download the required packages.

