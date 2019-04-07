# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

from .spider_lib import *

if __name__ == '__main__':
    res = conf_request("https://www.douban.com/")
    urls = get_all_url(res.text)
    ress = dyn_ua_requests(urls)
    for i, res in enumerate(ress):
        print(i, urls[i], get_all_text(res.text))
        exit(0)
