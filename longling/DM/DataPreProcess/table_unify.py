# coding: utf-8
# create by tongshiwei on 2019/5/26

"""
针对大规模数据场景，适用于单机对象

# table_field_unify.* 方法
包括两个功能：
    * 去除不想要的字段
    * 对剩余字段进行过滤
"""


def table_field_unify_with_header(iterable, output, unifier=None,
                                  includes=None, excludes=None):
    """
    Parameters
    ----------
    iterable: iterable
        待处理表数据
    output:
        输出，需要有 add 方法
    unifier: dict or list
    includes: set or list
        保留字段
    excludes: set
        不保留字段
    """
    header = next(iterable)
    if excludes is None:
        excludes = set()
    if unifier is None:
        unifier = {}

    header2index = dict(zip(header, range(len(header))))

    if includes is not None:
        for name in set(header2index) - set(includes):
            excludes.add(header2index[name])

    unifier2index = {
        header2index[name]: unifier[name]
        for name in unifier
    }
    output.add([name for name in header if header2index[name] not in excludes])
    table_field_unify(iterable, output, unifier2index, excludes)


def table_field_unify_without_header(iterable, output,
                                     unifier=None, excludes=None):
    """
    Parameters
    ----------
    iterable: iterable
        待处理表数据
    output:
        输出，需要有 add 方法
    unifier: dict or list
    excludes: set or dict
        不保留字段
    """
    return table_field_unify(iterable, output, unifier, excludes)


def table_field_unify(iterable, output, unifier=None, excludes=None):
    """
    Parameters
    ----------
    iterable: iterable
        待处理表数据
    output:
        输出，需要有 add 方法
    unifier: dict or list
    excludes: set or dict
        不保留字段
    """
    assert hasattr(output, "add")
    assert unifier or excludes
    excludes = set() if excludes is None else excludes
    unifier = {} if unifier is None else unifier
    assert not set(unifier) & set(excludes)
    for elems in iterable:
        _elems = []
        for i, elem in enumerate(elems):
            _elem = unifier[i](elem) if i in unifier else elem
            if i not in excludes:
                _elems.append(_elem)
        output.add(_elems)
