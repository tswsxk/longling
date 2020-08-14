# coding: utf-8
# 2019/12/10 @ tongshiwei

"""
Spider parser, targeting at html, parse the component
"""

from bs4 import BeautifulSoup


def as_bs4(lxml_text: (str, BeautifulSoup)) -> BeautifulSoup:
    return lxml_text if isinstance(lxml_text, BeautifulSoup) else BeautifulSoup(lxml_text)


__all__ = ["get_all_url"]


def get_all_url(lxml_text) -> list:
    return [a.get('href') for a in as_bs4(lxml_text).find_all('a')]


def get_all_text(lxml_text) -> str:
    return as_bs4(lxml_text).get_text()
