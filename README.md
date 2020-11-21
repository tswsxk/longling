# longling

[![Documentation Status](https://readthedocs.org/projects/longling/badge/?version=latest)](https://longling.readthedocs.io/zh/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/longling.svg)](https://pypi.python.org/pypi/longling)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/longling.svg)](https://pypi.python.org/pypi/longling)
[![Build Status](https://www.travis-ci.org/tswsxk/longling.svg?branch=master)](https://www.travis-ci.org/tswsxk/longling)
[![codecov](https://codecov.io/gh/tswsxk/longling/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/longling)
[![Download](https://img.shields.io/pypi/dm/longling.svg?style=flat)](https://pypi.python.org/pypi/longling)

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


## Developer Guidance

### commit format

```
<type>(<scope>): <subject>
```

#### type
feat：新功能（feature）。

fix/to：修复bug，可以是QA发现的BUG，也可以是研发自己发现的BUG。

* fix：产生diff并自动修复此问题。适合于一次提交直接修复问题
* to：只产生diff不自动修复此问题。适合于多次提交。最终修复问题提交时使用fix

docs：文档（documentation）。

style：格式（不影响代码运行的变动）。

refactor：重构（即不是新增功能，也不是修改bug的代码变动）。

perf：优化相关，比如提升性能、体验。

test：增加测试。

chore：构建过程或辅助工具的变动。

revert：回滚到上一个版本。

merge：代码合并。

sync：同步主线或分支的Bug。

#### scope(可选)

scope用于说明 commit 影响的范围，比如数据层、控制层、视图层等等，视项目不同而不同。

例如在Angular，可以是location，browser，compile，compile，rootScope， ngHref，ngClick，ngView等。如果你的修改影响了不止一个scope，你可以使用*代替。

#### subject(必须)

subject是commit目的的简短描述，不超过50个字符。

结尾不加句号或其他标点符号。
