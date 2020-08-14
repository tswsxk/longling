# coding: utf-8
# 2019/12/31 @ tongshiwei

import pytest

from longling.lib.regex import variable_replace, default_variable_replace


# from longling.lib.regex import PatternHitter
# from longling.lib.regex.pattener import mode_dict, register
# from longling import path_append, wf_open, rf_open


def test_variable_replace():
    demo_string1 = "hello $who, I am $name"

    with pytest.raises(KeyError):
        variable_replace(demo_string1, who="world")

    demo_string2 = "hello $WHO, I am $NAME"

    with pytest.raises(KeyError):
        variable_replace(demo_string2, key_lower=False, who="world", name="longling")

    with pytest.raises(TypeError):
        default_variable_replace("I'm $who", default_value={"groot"})

    assert variable_replace("$PROJECT-", project="longling") == "longling-"

#
# def test_patterner():
#     tmp = mode_dict[0]
#
#     @register(0)
#     def _init_patterns0(line, pps=None):
#         return mode_dict[0](line, pps)
#
#     mode_dict[0] = tmp
#
#
# def test_pattern_hitter(tmpdir):
#     regex_txt = path_append(tmpdir, "process_pattern_regex.txt", to_str=True)
#     regex_string = ":世界\n:和平"
#     with wf_open(regex_txt) as wf:
#         print(regex_string, file=wf)
#     p = PatternHitter(regex_txt, mode=1)
#     assert p.hit_num("世界 和平") == 2
#
#     assert p.hit_group("世界 和平") == [":世界", ":和平"]
#
#     for mode in mode_dict:
#         with pytest.raises(ValueError):
#             PatternHitter("1:2:3", mode=mode)
#
#     with pytest.raises(KeyError):
#         PatternHitter("1:2", mode=2)
