# coding: utf-8
# create by tongshiwei on 2019-8-27

import functools

import fire

from longling import Architecture as arch_cli
from longling.Architecture.cli import cli as arch
from longling.Architecture.install_file import nni as install_nni
from longling.ML import embedding_dim
from longling.ML.toolkit.analyser.cli import select_max, arg_select_max, select_min, arg_select_min
from longling.ML.toolkit.analyser.to_board import to_board
from longling.ML.toolkit.dataset import train_valid_test, train_test, kfold
from longling.ML.toolkit.hyper_search.nni import show_top_k, show
from longling.lib.loading import csv2jsonl, jsonl2csv
from longling.lib.stream import encode
from longling.spider import download_file as download
from longling.toolbox import toc

tarch = functools.partial(arch, author="Shiwei Tong")


def cli():  # pragma: no cover
    fire.Fire(
        {
            "train_valid_test": train_valid_test,
            "train_test": train_test,
            "kfold": kfold,
            "toc": toc,
            "arch": arch,
            "tarch": tarch,
            "arch-cli": arch_cli,
            "max": select_max,
            "amax": arg_select_max,
            "min": select_min,
            "amin": arg_select_min,
            "csv2jsonl": csv2jsonl,
            "jsonl2csv": jsonl2csv,
            "encode": encode,
            "download": download,
            "nni": {
                "k_best": show_top_k,
                "show": show,
            },
            "install": {
                "nni": install_nni,
            },
            "to_board": to_board,
            "ml": {
                "embedding_dim": embedding_dim,
            }
        }
    )


if __name__ == '__main__':
    cli()
