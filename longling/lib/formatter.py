# coding: utf-8
# 2020/4/13 @ tongshiwei

__all__ = ["dict_format", "pandas_format", "table_format", "series_format"]

from contextlib import contextmanager

from longling.lib.candylib import as_ordered_dict


def dict_format(data: dict, digits=6, col: int = None):
    """
    Examples
    --------
    >>> print(dict_format({"a": 123, "b": 3, "c": 4, "d": 5}))  # doctest: +NORMALIZE_WHITESPACE
    a: 123	b: 3	c: 4	d: 5
    >>> print(dict_format({"a": 123, "b": 3, "c": 4, "d": 5}, col=3))  # doctest: +NORMALIZE_WHITESPACE
    a: 123	b: 3	c: 4
    d: 5
    """
    if col is None:
        msg = []
        for name, value in data.items():
            if isinstance(value, float):
                _msg = "{}: {:.{digits}f}".format(name, value, digits=digits)
            else:
                _msg = "{}: {}".format(name, value)
            msg.append(_msg)
        msg = "\t".join([m for m in msg if m])
    else:
        msg = ""
        for i, (name, value) in enumerate(data.items()):
            if isinstance(value, float):
                _msg = "{}: {:.{digits}f}".format(name, value, digits=digits)
            else:
                _msg = "{}: {}".format(name, value)
            if (i + 1) % col == 0 and i != len(data) - 1:
                _msg += "\n"
            elif i != len(data) - 1:
                _msg += "\t"
            msg += _msg
    return msg


def pandas_format(data: (dict, list, tuple), columns: list = None, index: (list, str) = None, orient="index",
                  pd_kwargs: dict = None, max_rows=80, max_columns=80, **kwargs):
    """

    Parameters
    ----------
    data: dict, list, tuple, pd.DataFrame
    columns : list, default None
        Column labels to use when ``orient='index'``. Raises a ValueError
        if used with ``orient='columns'``.
    index : list of strings
        Optional display names matching the labels (same order).
    orient : {'columns', 'index'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
    pd_kwargs: dict
    max_rows: (int, None), default 80
    max_columns: (int, None), default 80

    Examples
    --------
    >>> print(pandas_format({"a": {"x": 1, "y": 2}, "b": {"x": 1.0, "y": 3}},  ["x", "y"]))
         x  y
    a  1.0  2
    b  1.0  3
    >>> print(pandas_format([[1.0, 2], [1.0, 3]],  ["x", "y"], index=["a", "b"]))
         x  y
    a  1.0  2
    b  1.0  3
    """

    import pandas as pd

    kwargs.update({
        "max_rows": max_rows,
        "max_columns": max_columns,
    })

    pd_kwargs = {} if pd_kwargs is None else pd_kwargs

    @contextmanager
    def pandas_session():
        for key, value in kwargs.items():
            pd.pandas.set_option(key, value)
        yield
        for key in kwargs:
            pd.pandas.reset_option(key)

    if isinstance(data, dict):
        data = as_ordered_dict(data, index)
        table = pd.DataFrame.from_dict(
            data, orient=orient, columns=columns, **pd_kwargs,
        )

    elif isinstance(data, (list, tuple, pd.DataFrame)):
        table = pd.DataFrame(data, index=index, columns=columns, **pd_kwargs)

    else:
        raise TypeError("cannot handle %s" % type(data))

    with pandas_session():
        return str(table)


series_format = dict_format
table_format = pandas_format
