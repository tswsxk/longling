# coding:utf-8
# created by tongshiwei on 2018/10/9

import os
import random

from tqdm import tqdm

from longling.lib.stream import wf_open, AddPrinter

__all__ = ["DatasetSplitter", "file_dataset_split"]


class DatasetSplitter(object):
    @staticmethod
    def split(iterable,
              valid_ratio=0.0, test_ratio=0.2,
              train_buffer=None, valid_buffer=None, test_buffer=None, silent=True):
        counter = {
            "train": 0,
            "valid": 0,
            "test": 0,
        }
        for element in tqdm(iterable, disable=silent):
            random_seed = random.random()
            if random_seed < test_ratio:
                buffer = test_buffer
            elif random_seed < valid_ratio + test_ratio:
                buffer = valid_buffer
            else:
                buffer = train_buffer
            buffer.add(element)

        return counter

    def __call__(self, iterable,
                 valid_ratio=0.0, test_ratio=0.2,
                 train_buffer=None, valid_buffer=None, test_buffer=None, silent=True):
        DatasetSplitter.split(
            iterable, valid_ratio, test_ratio,
            train_buffer, valid_buffer, test_buffer,
            silent
        )


def file_dataset_split(filename, valid_ratio=0.0, test_ratio=0.5, root_dir=None,
                       train_filename=None, valid_filename=None, test_filename=None,
                       silent=True):
    if train_filename or test_filename or valid_filename:
        assert train_filename and valid_filename and test_filename
    else:
        train_filename = filename + ".train"
        valid_filename = filename + ".valid"
        test_filename = filename + ".test"

    if root_dir:
        train_filename = os.path.join(root_dir, train_filename)
        test_filename = os.path.join(root_dir, test_filename)

    splitter = DatasetSplitter()

    with open(filename) as f, \
            wf_open(train_filename) as train_wf, \
            wf_open(valid_filename) as valid_wf, \
            wf_open(test_filename) as test_wf:

        train_buffer = AddPrinter(train_wf)
        valid_buffer = AddPrinter(valid_wf)
        test_buffer = AddPrinter(test_wf)

        counter = splitter(
            f, valid_ratio, test_ratio,
            train_buffer, valid_buffer, test_buffer,
            silent,
        )

    return counter
