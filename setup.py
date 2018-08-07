# coding: utf-8
# created by tongshiwei on 17-12-17
import io
import re
import os
from distutils.core import setup
from setuptools import find_packages

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
    packages=find_packages(
        include=[
            "__init__.py"
            "*.lib", "*.lib.*",
            "*.framework.ML.MXnet", "*.framework.ML.MXnet.*"
            "*.framework.ML.universe", "*.framework.ML.universe.*",
        ],
        exclude=[
            "*mx_example", "*gluon_example*", "*gluon_exp*",
            "*mxnet_old*",
        ]
    ),
    scripts=[
        # todo, 添加 glue 命令
    ],
    url='https://gitlab.com/tswsxk/longling.git',
    license='LICENSE.txt',
    description='handy wrapper for many libs',
    long_description=open('README.md').read(),
    install_requires=[
        "pip"
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

)
