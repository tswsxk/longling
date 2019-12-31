# coding: utf-8
# 2019/12/31 @ tongshiwei

import pytest
from longling.lib.process_pattern import variable_replace


def test_variable_replace():
    demo_string1 = "hello $who, I am $name"

    with pytest.raises(KeyError):
        variable_replace(demo_string1, who="world")

    demo_string2 = "hello $WHO, I am $NAME"

    with pytest.raises(KeyError):
        variable_replace(demo_string2, key_lower=False, who="world", name="longling")
