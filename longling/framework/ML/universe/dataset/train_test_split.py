# coding:utf-8
# created by tongshiwei on 2018/10/9

import os
from longling.lib.stream import wf_open
from tqdm import tqdm
import random


def train_test_split(filename, test_ratio=0.5, root_dir=None, train_filename=None, test_filename=None,
                     progress_viz=True):
    if train_filename or test_filename:
        assert train_filename and test_filename
    else:
        train_filename = filename + ".train"
        test_filename = filename + ".test"

    if root_dir:
        train_filename = os.path.join(root_dir, train_filename)
        test_filename = os.path.join(root_dir, test_filename)

    with open(filename) as f, wf_open(train_filename) as train_wf, wf_open(test_filename) as test_wf:
        for line in tqdm(f, disable=not progress_viz):
            if random.random() < test_ratio:
                wf = test_wf
            else:
                wf = train_wf
            print(line, end='', file=wf)
