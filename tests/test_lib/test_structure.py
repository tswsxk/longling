# coding: utf-8
# 2019/12/19 @ tongshiwei

import pytest
from longling.lib.structure import AttrDict


def test_attr_dict_exception():
    ad = AttrDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    del ad.age

    with pytest.raises(AttributeError):
        del ad.age

    with pytest.raises(AttributeError):
        print(ad.age)

    ad.age = 10

    del ad["age"]

    with pytest.raises(KeyError):
        del ad["age"]

    with pytest.raises(KeyError):
        print(ad["age"])
