# coding: utf-8
# created by tongshiwei on 17-12-17
import io
import os
import re
from distutils.core import setup

from setuptools import find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-pep8>=1',
]

doc_deps = [
    'sphinx',
    'sphinx-rtd-theme',
    'recommonmark'
]

dev_deps = test_deps + doc_deps + [
    'setuptools>=40',
    'wheel'
]

ml_base_deps = [
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
]

ml_mx_deps = [
    "mxnet",
    "gluonnlp",
]
ml_pytorch_deps = [
    "torch"
]

ml_full_deps = ml_base_deps + ml_mx_deps + ml_pytorch_deps


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
            "*.Architecture", "*.Architecture.*",
            "*.ML",
            "*.ML.DL*",
            "*.ML.MxnetHelper*",
            "*.ML.PytorchHelper*",
            "*.ML.toolkit*",
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
    ],
    extras_require={
        'test': test_deps,
        'doc': doc_deps,
        'dev': dev_deps,
        'ml': ml_base_deps,
        'ml-full': ml_full_deps,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
