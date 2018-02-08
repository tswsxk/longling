# coding: utf-8

from __future__ import print_function

import sys

if sys.version_info[0] == 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

from longling.lib.stream import check_dir, check_file

import logging


def url_download(durl, filename=None, dirname="", reload=False):
    if filename is None or not filename:
        filename = durl.split("/")[-1]
    filename = dirname + filename
    check_dir(filename)
    if check_file(filename) and not reload:
        logging.debug("url_download-file exists")
        return 0
    status = urlretrieve(durl, filename)
    return status


if __name__ == '__main__':
    print(url_download("https://pic3.zhimg.com/daddeafad987ac0a44e546af76a3ca6a_xs.jpg"))
