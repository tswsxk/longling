# coding: utf-8
# 2020/5/11 @ tongshiwei


def key_parser(key):
    """
    Examples
    --------
    >>> key_parser("macro avg:f1")
    ['macro avg', 'f1']
    >>> key_parser("accuracy")
    'accuracy'
    >>> key_parser("iteration:accuracy")
    ['iteration', 'accuracy']
    """
    if ":" in key:
        # prf:0:f1
        return key.split(":")
    return key


def get_by_key(data, parsed_key):
    _key = key_parser(parsed_key)
    _data = data
    if isinstance(_key, list):
        for k in _key:
            _data = _data[k]
    else:
        _data = _data[_key]
    return _data
