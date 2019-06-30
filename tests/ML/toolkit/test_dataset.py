# coding: utf-8
# create by tongshiwei on 2019/6/27

from tests.lib.stream import NullDevice
from longling.ML.toolkit.dataset import file_dataset_split, DatasetSplitter


def test_dataset_split():
    pesudo_data = list(range(0, 1000))
    spliter = DatasetSplitter()
    spliter(
        pesudo_data,
        train_buffer=NullDevice(), valid_buffer=NullDevice(), test_buffer=NullDevice(),
        silent=False,
    )
