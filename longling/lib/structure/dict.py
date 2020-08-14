# coding: utf-8
# create by tongshiwei on 2019/4/8

__all__ = ["AttrDict", "nested_update"]


class AttrDict(dict):
    """
    Example
    -------
    >>> ad = AttrDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    >>> ad
    {'first_name': 'Eduardo', 'last_name': 'Pool', 'age': 24, 'sports': ['Soccer']}
    >>> ad.first_name
    'Eduardo'
    >>> ad.age
    24
    >>> ad.age = 16
    >>> ad.age
    16
    >>> ad["age"] = 20
    >>> ad["age"]
    20
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(str(e))

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.__delitem__(item)
        except KeyError as e:
            raise AttributeError(str(e))

    def __setitem__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delitem__(self, key):
        super(AttrDict, self).__delitem__(key)
        del self.__dict__[key]


def nested_update(src: dict, update: dict):
    """
    Examples
    -------
    >>> nested_update({"a": {"x": 1}}, {"a": {"y": 2}})
    {'a': {'x': 1, 'y': 2}}
    >>> nested_update({"a": {"x": 1}}, {"a": {"x": 2}})
    {'a': {'x': 2}}
    >>> nested_update({"a": {"x": 1}}, {"b": {"y": 2}})
    {'a': {'x': 1}, 'b': {'y': 2}}
    >>> nested_update({"a": {"x": 1}}, {"a": 2})
    {'a': 2}
    """
    if isinstance(update, dict):
        for key, value in update.items():
            if key in src and isinstance(src[key], dict):
                if isinstance(value, dict):
                    nested_update(src[key], value)
                else:
                    src[key] = value
            else:
                src[key] = value
    return src
