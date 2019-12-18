# coding: utf-8
# 2019/12/10 @ tongshiwei

from bs4 import BeautifulSoup
import requests

from longling.spider.parser import lxml


def test_lxml_parser():
    url = "http://base.ustc.edu.cn/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    lxml_text = BeautifulSoup(r.text, "lxml")

    assert "http://bigdata.ustc.edu.cn/" in set(lxml.get_all_url(lxml_text))

    lxml.get_all_text(lxml_text)

    assert True
