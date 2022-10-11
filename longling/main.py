# coding: utf-8
# create by tongshiwei on 2019-8-27

import functools

import fire

from longling import Architecture as ArchCli
from longling.Architecture.cli import cli as arch
from longling.Architecture.install_file import nni as install_nni
from longling.lib.loading import csv2jsonl, jsonl2csv
from longling.lib.stream import encode
from longling.toolbox import toc

tarch = functools.partial(arch, author="Shiwei Tong")


def embedding_dim_cli(embedding_size):
    from longling.ML import embedding_dim
    embedding_dim(embedding_size)


def select_max_cli(src, *keys, with_keys=None, with_all=False, **kwargs):
    from longling.ML.toolkit.analyser.cli import select_max
    select_max(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def arg_select_max_cli(*keys, src, with_keys=None, with_all=False, **kwargs):
    from longling.ML.toolkit.analyser.cli import arg_select_max
    arg_select_max(*keys, src, with_keys=with_keys, with_all=with_all, **kwargs)


def select_min_cli(src, *keys, with_keys=None, with_all=False, **kwargs):
    from longling.ML.toolkit.analyser.cli import select_min
    select_min(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def arg_select_min_cli(*keys, src, with_keys=None, with_all=False, **kwargs):
    from longling.ML.toolkit.analyser.cli import arg_select_min
    arg_select_min(src, *keys, with_keys=with_keys, with_all=with_all, **kwargs)


def to_board_cli(src, board_dir, global_step_field, *scalar_fields):
    from longling.ML.toolkit.analyser.to_board import to_board
    to_board(src, board_dir, global_step_field, *scalar_fields)


def train_valid_test_cli(*files,
                         train_size: (float, int) = 0.8, valid_size: (float, int) = 0.1,
                         test_size: (float, int, None) = None, ratio=None,
                         random_state=None,
                         shuffle=True, target_names=None,
                         suffix: list = None, prefix="", logger=None, **kwargs):
    from longling.ML.toolkit.dataset import train_valid_test
    train_valid_test(
        *files,
        train_size, valid_size, test_size,
        ratio,
        random_state,
        shuffle,
        target_names,
        suffix,
        prefix,
        logger,
        **kwargs
    )


def train_test_cli(*files, train_size: (float, int) = 0.8, test_size: (float, int, None) = None, ratio=None,
                   random_state=None, shuffle=True, target_names=None,
                   suffix: list = None, prefix="", logger=None, **kwargs):
    from longling.ML.toolkit.dataset import train_test
    train_test(
        *files,
        train_size, test_size,
        ratio, random_state, shuffle, target_names, suffix, prefix,
        logger, **kwargs
    )


def kfold_cli(*files, n_splits=5, shuffle=False, random_state=None, suffix=None, prefix="", logger=None):
    from longling.ML.toolkit.dataset import kfold
    kfold(*files, n_splits, shuffle, random_state, suffix, prefix, logger)


def show_top_k_cli(k, exp_id=None, exp_dir=None):
    from longling.ML.toolkit.hyper_search.nni import show_top_k
    show_top_k(k, exp_id, exp_dir)


def show_cli(key, max_key=True, exp_id=None, res_dir="./",
             nni_dir=None,
             only_final=False,
             with_keys=None, with_all=False):
    from longling.ML.toolkit.hyper_search.nni import show
    show(key, max_key, exp_id, res_dir, nni_dir, only_final, with_keys, with_all)


def download_cli(url, save_path=None, override=True, decomp=True, reporthook=None):
    from longling.spider import download_file as download
    download(url, save_path, override, decomp, reporthook)


def cli():  # pragma: no cover
    fire.Fire(
        {
            "train_valid_test": train_valid_test_cli,
            "train_test": train_test_cli,
            "kfold": kfold_cli,
            "toc": toc,
            "arch": arch,
            "tarch": tarch,
            "arch-cli": ArchCli,
            "max": select_max_cli,
            "amax": arg_select_max_cli,
            "min": select_min_cli,
            "amin": arg_select_min_cli,
            "csv2jsonl": csv2jsonl,
            "jsonl2csv": jsonl2csv,
            "encode": encode,
            "download": download_cli,
            "nni": {
                "k_best": show_top_k_cli,
                "show": show_cli,
            },
            "install": {
                "nni": install_nni,
            },
            "to_board": to_board_cli,
            "ml": {
                "embedding_dim": embedding_dim_cli,
            }
        }
    )


if __name__ == '__main__':
    cli()
