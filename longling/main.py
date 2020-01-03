# coding: utf-8
# create by tongshiwei on 2019-8-27

import functools
import fire

from longling.lib.stream import encoding
from longling.ML.toolkit.dataset import train_valid_test, train_test, train_valid, kfold
from longling.ML.toolkit.analyser.cli import select_max, arg_select_max
from longling.toolbox import toc
from longling.Architecture.cli import cli as arch
from longling import Architecture as arch_cli
from longling.lib.loading import csv2json, json2csv

tarch = functools.partial(arch, author="Shiwei Tong")


def cli():  # pragma: no cover
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
            "max": select_max,
            "amax": arg_select_max,
            "csv2json": csv2json,
            "json2csv": json2csv,
            "encoding": encoding,
        }
    )


if __name__ == '__main__':
    cli()
