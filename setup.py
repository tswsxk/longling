# coding: utf-8
# created by tongshiwei on 17-12-17
import logging
import io
import os
import re
from distutils.core import setup

from setuptools import find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

docs_deps = [
    'sphinx',
    'sphinx_rtd_theme',
    'nbsphinx',
    'm2r2',
    'Image',
    'recommonmark',
    'ipython',
    'sphinx_toggleprompt'
]

dev_deps = test_deps + docs_deps + [
    'setuptools>=40',
    'wheel',
    'twine'
]

dm_base_deps = [
    "pandas",
    "numpy",
    "matplotlib",
]

ml_base_deps = [
    "pandas",
    "numpy>= 1.16.5",
    "scipy",
    "scikit-learn>=0.23.2",
]
viz_deps = [
    "matplotlib",
    "tensorboardx",
    "tensorboard"
]

try:
    import mxnet

    mxnet_requires = []
except ModuleNotFoundError:
    mxnet_requires = ["mxnet"]
except Exception as e:
    mxnet_requires = []
    logging.error(e)

try:
    import torch

    ml_pytorch_deps = []
except ModuleNotFoundError:
    import sys

    if 5 <= sys.version_info[1]:
        ml_pytorch_deps = ["torch"]
    else:
        ml_pytorch_deps = []
        logging.warning("Current python version %s is not supported by pytorch", str(sys.version_info[:2]))
except Exception as e:
    ml_pytorch_deps = []
    logging.error(e)

ml_mx_deps = ["gluonnlp"] + mxnet_requires

spider_deps = [
    "requests",
    "rarfile",
    "bs4",
    "lxml",
    "urllib3"
]

dl_deps = ml_mx_deps + ml_pytorch_deps

ml_full_deps = ml_base_deps + dl_deps + ["nni>=1.8"] + viz_deps

full_deps = ml_full_deps + spider_deps


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('longling', '__init__.py')

setup(
    name='longling',
    version=VERSION,
    author='Sherlock, Shiwei Tong',
    author_email='tongsw@mail.ustc.edu.cn',
    include_package_data=True,
    python_requires='>=3.6',
    packages=find_packages(
        include=[
            "longling",
            "*.toolbox", "*.toolbox.*",
            "*.lib", "*.lib.*",
            "*.spider", "*.spider.*",
            "*.Architecture", "*.Architecture.*",
            "*.ML*",
        ],
        exclude=[
            "*.mx_example", "*.gluon_example*", "*.gluon_exp*",
            "*.mxnet_old*",
        ]
    ),
    entry_points={
        "console_scripts": [
            "glue = longling.ML.MxnetHelper.glue.glue:cli",
            "longling = longling.main:cli"
        ],
    },
    url='https://gitlab.com/tswsxk/longling.git',
    license='LICENSE.txt',
    description='This project aims to provide some handy toolkit functions to help construct the architecture.',
    long_description=open('README.txt', encoding="utf-8").read(),
    install_requires=[
        "pip",
        "tqdm",
        "fire",
        "PyYAML",
        "toml",
    ],
    extras_require={
        'test': test_deps + full_deps,
        'doc': docs_deps,
        'dev': dev_deps,
        'dm': dm_base_deps,
        'ml': ml_base_deps,
        'ml-viz': ml_base_deps + viz_deps,
        'viz': viz_deps,
        'mx': ml_mx_deps,
        'torch': ml_pytorch_deps,
        'dl': dl_deps,
        'ml-full': ml_full_deps,
        "spider": spider_deps,
        "full": full_deps
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
