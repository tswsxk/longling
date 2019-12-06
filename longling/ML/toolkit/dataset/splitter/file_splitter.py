# coding: utf-8
# create by tongshiwei on 2019-8-27

__all__ = [
    "file_ratio_split",
    "train_valid_test", "train_test", "train_valid",
    "kfold"
]
from longling.lib.parser import Formatter, path_append
from longling.lib.stream import wf_open, AddPrinter

from .iterable_splitter import RatioSplitter, KFoldSplitter


def file_ratio_split(filename_list, ratio_list, root_dir=None, filename_formatter_list=None):
    assert len(ratio_list) == len(filename_formatter_list)
    if root_dir is not None:
        filename_formatter_list = [path_append(root_dir, f) for f in filename_formatter_list]
    wfps = []
    fps = []
    source_target_iterable = []
    for filename in filename_list:
        _wfps = [
            wf_open(Formatter.format(filename, formatter=formatter)) for formatter in filename_formatter_list
        ]
        wfps.append(_wfps)
        targets = [
            AddPrinter(wfp, end='') for wfp in _wfps
        ]
        fp = open(filename, encoding="utf-8")
        fps.append(fp)
        source_target_iterable.append((fp, targets))

    splitter = RatioSplitter(fps[0], *ratio_list)
    splitter.split(*source_target_iterable)

    for _wfps in wfps:
        for wfp in _wfps:
            wfp.close()
    for fp in fps:
        fp.close()


def train_valid_test(*filename,
                     train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2,
                     root_dir=None,
                     train_filename="{}.train", valid_filename="{}.valid", test_filename="{}.test"):
    file_ratio_split(
        filename,
        [train_ratio, valid_ratio, test_ratio],
        root_dir,
        [train_filename, valid_filename, test_filename]
    )


def train_valid(*filename: str,
                train_ratio: float = 0.8, valid_ratio: float = 0.2,
                root_dir: str = None,
                train_filename="{}.train", valid_filename="{}.valid"):
    """
    1. filename 可以是一个或多个字符串，指明待切分的文件名，可以通过指定root_dir来批量增加路径前缀

    2. train_filename 和 valid_filename 都是通过字符串format方式生成的:
    $train_filename = train_filename.format($filename),
    其中 $train_filename 指的是生成并使用存储训练数据的文件文件名，
    而 $filename 指的是第一步中合并root_dir之后的源文件文件名

    Parameters
    ----------
    filename: str or list
        需要进行划分的源文件
    train_ratio
    valid_ratio
    root_dir: str
        根目录
    train_filename:

    valid_filename

    Returns
    -------

    """
    file_ratio_split(
        filename,
        [train_ratio, valid_ratio],
        root_dir,
        [train_filename, valid_filename]
    )


def train_test(*filename,
               train_ratio=0.8, test_ratio=0.2,
               root_dir=None,
               train_filename="{}.train", test_filename="{}.test"):
    file_ratio_split(
        filename,
        [train_ratio, test_ratio],
        root_dir,
        [train_filename, test_filename]
    )


def kfold(*filename, n_splits, root_dir=None, train_filename="{}_{}.train", test_filename="{}_{}.test"):
    if root_dir is not None:
        train_filename = path_append(root_dir, train_filename)
        test_filename = path_append(root_dir, test_filename)

    target_filename_list = [
        [
            [Formatter.format(_filename, i, train_filename) for i in range(n_splits)],
            [Formatter.format(_filename, i, test_filename) for i in range(n_splits)]
        ]
        for _filename in filename
    ]

    wfps = []
    fps = []
    source_target_iterable = []
    for _source_target_iterable in zip(filename, target_filename_list):
        _wfps = [
            wf_open(_wf) for _wf in _source_target_iterable[1]
        ]
        targets = [
            AddPrinter(fp) for fp in _wfps
        ]
        wfps.append(_wfps)
        fp = open(_source_target_iterable[0], encoding="utf-8")
        fps.append(fp)
        source_target_iterable.append((fp, targets))

    splitter = KFoldSplitter(fps[0], n_splits)
    splitter.split(*source_target_iterable)

    for _wfps in wfps:
        for wfp in _wfps:
            wfp.close()
    for fp in fps:
        fp.close()
