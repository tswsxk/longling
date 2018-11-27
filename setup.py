# coding: utf-8
# created by tongshiwei on 17-12-17
import io
import re
import os
# import sys
from distutils.core import setup
from setuptools import find_packages
from longling.framework.ML.MXnet.mx_gluon.glue import glue


# CURRENT_DIR = os.path.dirname(__file__)
# sys.path.insert(0, CURRENT_DIR)

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
            "longling",
            "*.lib", "*.lib.*",
            "*.framework",
            "*.framework.ML",
            "*.framework.ML.MXnet*",
            "*.framework.ML.universe*"
        ],
        exclude=[
            "*.mx_example", "*.gluon_example*", "*.gluon_exp*",
            "*.mxnet_old*",
        ]
    ),
    scripts=[
        glue.__file__,
    ],
    url='https://gitlab.com/tswsxk/longling.git',
    license='LICENSE.txt',
    description='handy wrapper for many libs',
    long_description=open('README.md').read(),
    install_requires=[
        "pip"
    ],
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
