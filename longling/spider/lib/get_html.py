# coding: utf-8
# 2020/3/23 @ tongshiwei

import urllib.request
from .utils import get_http_header, get_opener
from longling.spider.conf import logger

__all__ = ["get_html_code"]


def get_html_code(url):
    """
    get encoded html code from specified url
    """
    logger.debug("request %s" % url)
    req = urllib.request.Request(url=url, headers=get_http_header(), method='GET')
    opener = get_opener()
    ret = opener.open(req)
    assert ret.status == 200, "error when request %s, error code %s" % (url, ret.status)
    content = ret.read()
    assert content, "error when request %s, null content" % url
    try:
        content = content.decode('utf-8')
    except UnicodeDecodeError:  # pragma: no cover
        content = content.decode('utf-8', 'ignore')
    return content
