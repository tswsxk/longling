# coding: utf-8
# 2020/4/16 @ tongshiwei

from pathlib import PurePath
import json
import math
import numpy as np
from longling import loading, as_out_io, PATH_TYPE, json_load, close_io
from longling import AddPrinter, type_from_name, config_logging
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import KFold
from longling import concurrent_pool

_logger = config_logging(logger="dataset", console_log_level="info")


def _get_s_points(length: int, ratio: list) -> list:
    """
    Returns
    -------
    s_points: list of int

    Examples
    --------
    >>> _get_s_points(10, [0.1, 0.9])
    [0, 1, 10]
    """
    cum_ratios = np.cumsum([0.0] + ratio)
    s_points = [
        math.ceil(length * ratio) for ratio in cum_ratios
    ]
    return s_points


def _get_s_indices(indices: list, ratio: list) -> list:
    """
    Returns
    -------
    s_indices: list of set of int

    Examples
    --------
    >>> _get_s_indices(list(range(10)), ratio=[0.1, 0.9])
    [{0}, {1, 2, 3, 4, 5, 6, 7, 8, 9}]
    """
    s_points = _get_s_points(len(indices), ratio)
    separator_indices = [
        set(indices[s_points[i]: s_points[i + 1]])
        for i in range(len(s_points) - 1)
    ]
    return separator_indices


def shuffle_indices(indices, random_state=None) -> list:
    """
    Examples
    --------
    >>> shuffle_indices(list(range(10)), 10)
    [8, 2, 5, 6, 3, 1, 0, 7, 4, 9]
    """
    return check_random_state(random_state).permutation(indices).tolist()


def _ratio_list(ratio: (str, list)) -> list:
    """
    Examples
    --------
    >>> _ratio_list("2:1:1")
    [0.5, 0.25, 0.25]
    >>> _ratio_list("0.5:0.25:0.25")
    [0.5, 0.25, 0.25]
    >>> _ratio_list([2, 1, 1])
    [0.5, 0.25, 0.25]
    >>> _ratio_list([0.5, 0.25, 0.25])
    [0.5, 0.25, 0.25]
    """
    if isinstance(ratio, str):
        ratio = list(map(float, ratio.split(":")))

    return (np.asarray(ratio) / np.sum(ratio)).tolist()


def get_s_indices(*files, ratio: (str, list) = None, index_file=None,
                  s_indices: (PATH_TYPE, list) = None, shuffle=True, random_state=None):
    if s_indices is not None:
        if isinstance(s_indices, PATH_TYPE):
            s_indices = [set(_s_indices) for _s_indices in json_load(s_indices)]

    elif ratio is not None:
        ratio = _ratio_list(ratio)

        src = zip(*[loading(_file) for _file in files])
        indices = [i for i, _ in enumerate(src)]

        if shuffle:
            indices = shuffle_indices(indices, random_state)

        s_indices = _get_s_indices(indices, ratio)
    else:
        raise ValueError("ratio or s_indices should be specified")

    if index_file is not None:
        with as_out_io(index_file) as wf:
            json.dump([list(_s_indices) for _s_indices in s_indices], wf)

    return s_indices


def __dump_file(src, s_indices: list, tars: list, logger=_logger):
    """

    Parameters
    ----------
    s_indices: list of set of int
    src: PATH_TYPE
    tars: list of AddPrinter
    logger

    """
    logger.debug("split and dump %s: start" % src)
    for i, elem in enumerate(loading(src)):
        for j, values in enumerate(s_indices):
            if i in values:
                tars[j].add(elem)
                break
        else:
            raise ValueError()
    logger.debug("split and dump %s: end" % src)


def _dump_file(srcs, s_indices: list, tars_list: list, logger=_logger):
    with concurrent_pool("t", 5) as pool:
        for src, tars in zip(srcs, tars_list):
            pool.submit(__dump_file, src, s_indices, tars, logger)


def _target_files(*files, target_names=None, suffix):
    return [
        [
            AddPrinter("%s%s" % (PurePath(_file).with_suffix(_suffix), type_from_name(_file))) if type_from_name(
                _file) in {".jsonl"}
            else AddPrinter("%s%s" % (PurePath(_file).with_suffix(_suffix), type_from_name(_file)), end='')
            for _suffix in suffix
        ] for _file in files
    ] if target_names is None else target_names


def _get_target_file_names(printers):
    return [printer.name for printer in printers]


def _src2tar_tips(srcs, tars, tips="", logger=_logger):
    for src, tar in zip(srcs, tars):
        logger.info("%s%s -> %s" % (tips, src, ",".join(_get_target_file_names(tar))))


def file_split(*files, ratio: (str, list) = None, target_files: list, index_file=None,
               s_indices: (PATH_TYPE, list) = None, shuffle=True, random_state=None, logger=_logger):
    """

    Parameters
    ----------
    files: list of PATH_TYPE
    ratio
    target_files: list of AddPrinter or list of list of AddPrinter
    index_file
    s_indices: PATH_TYPE or list of set of int

    Returns
    -------

    """
    s_indices = get_s_indices(*files, ratio=ratio, index_file=index_file, s_indices=s_indices, shuffle=shuffle,
                              random_state=random_state)

    _dump_file(files, s_indices, target_files, logger=logger)


def train_test(*files, train_size: (float, int) = 0.8, test_size: (float, int, None) = None, random_state=None,
               shuffle=True, target_names=None,
               suffix: list = None, **kwargs):
    """

    Parameters
    ----------
    files
    train_size: float, int, or None, (default=0.8)
        Represent the proportion of the dataset to include in the train split.
    test_size: float, int, or None
        Represent the proportion of the dataset to include in the train split.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.
    shuffle: boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None
    target_names: list of PATH_TYPE
    suffix: list
    kwargs
    """
    suffix = [".train", ".test"] if suffix is None else suffix

    assert len(suffix) == 2

    target_files = _target_files(*files, target_names=target_names, suffix=suffix)

    ratio = "%s:" % train_size

    if test_size is None:
        assert 0 <= train_size <= 1
        test_size = "%s" % (1 - train_size)
    else:
        test_size = "%s" % test_size

    ratio += test_size

    file_split(
        *files,
        ratio=ratio,
        target_files=target_files,
        shuffle=shuffle,
        random_state=random_state,
        **kwargs
    )
    close_io([[printer for printer in printers] for printers in target_files])


def train_valid_test(*files,
                     train_size: (float, int) = 0.8, valid_size: (float, int) = 0.1,
                     test_size: (float, int, None) = None,
                     random_state=None,
                     shuffle=True, target_names=None,
                     suffix: list = None, logger=_logger, **kwargs):
    """

    Parameters
    ----------
    files
    train_size: float, int, or None, (default=0.8)
        Represent the proportion of the dataset to include in the train split.
    valid_size: float, int, or None, (default=0.1)
        Represent the proportion of the dataset to include in the valid split.
    test_size: float, int, or None
        Represent the proportion of the dataset to include in the test split.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.
    shuffle: boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None
    target_names
    suffix: list
    kwargs
    """
    suffix = [".train", ".valid", ".test"] if suffix is None else suffix

    assert len(suffix) == 3

    target_files = _target_files(*files, target_names=target_names, suffix=suffix)

    ratio = "%s:%s:" % (train_size, valid_size)

    if test_size is None:
        assert 0 <= train_size + valid_size <= 1
        test_size = "%s" % (1 - train_size - valid_size)
    else:
        test_size = "%s" % test_size

    ratio += test_size

    file_split(
        *files,
        ratio=ratio,
        target_files=target_files,
        shuffle=shuffle,
        random_state=random_state,
        **kwargs
    )
    close_io([[printer for printer in printers] for printers in target_files])


def _kfold(*files, idx, indices, suffix, logger=_logger):
    logger.info("kfold %s start" % idx)
    s_indices = [set(_indices.tolist()) for _indices in indices]
    target_files = _target_files(
        *files, target_names=None, suffix=[".%s" % idx + _suffix for _suffix in suffix]
    )
    _src2tar_tips(files, target_files, "kfold %s: " % idx)
    file_split(*files, s_indices=s_indices, target_files=target_files, logger=logger)
    close_io([[printer for printer in printers] for printers in target_files])
    logger.info("kfold %s end" % idx)


def kfold(*files, n_splits=5, shuffle=False, random_state=None, suffix=None, logger=_logger):
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    suffix = [".train", ".test"] if suffix is None else suffix

    with concurrent_pool("t") as pool:
        for i, indices in enumerate(splitter.split(range(len([_ for _ in loading(files[0])])))):
            pool.submit(
                _kfold,
                *files,
                idx=i,
                indices=indices,
                suffix=suffix,
                logger=logger,
            )
