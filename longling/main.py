# coding: utf-8
# create by tongshiwei on 2019-8-27

import fire

from longling.ML.toolkit.dataset import train_valid_test, train_test, train_valid, kfold
from longling.toolbox import toc

def cli():
    fire.Fire(
        {
            "train_valid_test": train_valid_test,
            "train_test": train_test,
            "train_valid": train_valid,
            "kfold": kfold,
            "toc": toc,
        }
    )


if __name__ == '__main__':
    cli()
