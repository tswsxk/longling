# coding: utf-8
# created by tongshiwei on 17-12-17
from distutils.core import setup
from setuptools import find_packages

setup(
    name='longling',
    version='0.0.1',
    author='Sherlock, Shiwei Tong',
    author_email='tongsw@mail.ustc.edu.cn',
    packages=find_packages(),
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='handy wrapper for many libs',
    long_description=open('README.txt').read(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software Foundation (ASF)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

)
