# coding: utf-8
# 2020/3/27 @ tongshiwei


from longling.spider.lib import get_html_code


def test_get_html_code():
    assert get_html_code("https://tswsxk.github.io")
