# coding: utf-8
# created by tongshiwei on 17-12-17
from distutils.core import setup
from setuptools import find_packages

setup(
    name='longling',
    version='0.0.11',
    author='Sherlock, Shiwei Tong',
    author_email='tongsw@mail.ustc.edu.cn',
    packages=find_packages(
        include=[
            "*.lib", "*.lib.*",
            "*.framework.ML.MXnet", "*.framework.ML.MXnet.*"
            "*.framework.ML.universe", "*.framework.ML.universe.*",
        ],
        exclude=[
            "*mx_example", "*gluon_example*", "*gluon_exp*",
            "*mxnet_old*",
        ]
    ),
    scripts=[],
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
