# coding: utf-8
# create by tongshiwei on 2019-8-27

import functools
import fire

from longling.ML.toolkit.dataset import train_valid_test, train_test, train_valid, kfold
from longling.toolbox import toc
from longling.Architecture import cli as arch
from longling import Architecture as arch_cli


def tarch():
    return functools.partial(arch, author="Shiwei Tong")


def cli():
    fire.Fire(
        {
            "train_valid_test": train_valid_test,
            "train_test": train_test,
            "train_valid": train_valid,
            "kfold": kfold,
            "toc": toc,
            "arch": arch,
            "tarch": tarch,
            "arch-cli": arch_cli,
        }
    )


if __name__ == '__main__':
    cli()
