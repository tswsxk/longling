from __future__ import absolute_import
from __future__ import division

import logging
import os

from collections import OrderedDict

from tqdm import tqdm
import requests

from longling.base import tostr
from longling.lib.utilog import config_logging
from longling.lib.stream import check_file, wf_open, wf_close
from longling.framework.spider import conf

logger = config_logging(logger="spider", console_log_level=logging.INFO, propagate=False)

COOKIES = conf.cookies
HEADERS = conf.headers


def conf_request(url, cookies=conf.cookies, headers=conf.headers):
    return requests.get(url, cookies=cookies, headers=headers)


def dyn_ua_requests(urls, cookies=conf.cookies, ua_list="doc/agent_list"):
    import random
    with open(ua_list) as f:
        uas = [line.strip() for line in f if '#' not in line]

    for url in urls:
        ua = uas[random.randint(0, len(uas) - 1)]
        headers = {
            'User-Agent': ua,
        }
        yield requests.get(url, cookies=cookies, headers=headers)


def url_download(url, filename="", dirname="", reload=False, unit=None, retry=10, file_type=None):
    unit_dict = OrderedDict(
        [
            ("B", 1024 ** 0,),
            ("KB", 1024 ** 1),
            ("MB", 1024 ** 2),
            ("GB", 1024 ** 3),
            ("TB", 1024 ** 4),
        ]
    )

    if filename is None or not filename:
        filename = url.split("/")[-1]
    filename = os.path.abspath(os.path.join(dirname, filename))

    try:
        r = requests.get(url, stream=True)
    except Exception as e:
        raise TypeError("%s requests failed due to %s" % (url, e))

    if not r.status_code == 200:
        raise TypeError("fail-status code:%s" % r.status_code)
    try:
        size = int(r.headers.get('content-length', None))
    except TypeError:
        raise TypeError("%s data not correct" % url)

    try:
        if file_type is not None:
            downlaod_file_type = r.headers.get('content-type').split('/')[-1]
            if file_type == "":
                filename += ("." + downlaod_file_type)
            elif downlaod_file_type != file_type:
                raise TypeError("%s file type not consist" % url)
    except TypeError:
        pass

    logger.info("url: %s, target: %s", url, filename)
    if check_file(filename, size):
        if not reload:
            logger.info("file exists, size consists, no reload")
            return {'status_code': 0, 'filename': filename}
        else:
            logger.info("file exists, reload")

    if unit is None:
        for u, value in unit_dict.items():
            if size // value < 1024:
                unit = u
                unit_base = value
                break
    else:
        unit_base = unit_dict[unit]

    if size is None:
        logger.info("downloading start: data from %s to %s", url, filename)
    else:
        logger.info("downloading start: %s %s data from %s to %s", size / unit_base, unit, url, filename)

    for _ in range(retry):
        try:
            wf = wf_open(filename, mode="wb")
            tins = tqdm(desc="downloading data from %s" % url, unit=unit, total=size / unit_base)
            bits_cnt = 0
            for buffer in r.iter_content(8192):
                if not buffer:
                    break
                bits_cnt += len(buffer) / unit_base
                tins.update(len(buffer) / unit_base)
                wf.write(buffer)
            file_size = os.path.getsize(filename) / unit_base
            if bits_cnt != size / unit_base:
                logger.warning("downloading does not complete, retry")
                tins.close()
                wf_close(wf)
                continue
            logger.info("downloading finished\n%s %s data from %s\n%s %s written to %s, ",
                        bits_cnt, unit, url, file_size, unit, filename)
            return {'status_code': 1, 'filename': filename}
        except Exception:
            r = requests.get(url, stream=True)
            if not r.status_code == 200:
                raise TypeError("fail-status code:%s" % r.status_code)
        finally:
            tins.close()
            wf_close(wf)
    raise EOFError("data not downloading entirely")


if __name__ == '__main__':
    url = "https://scholar.google.com/citations?user=5EoHAFwAAAAJ&hl=en"
    r = requests.get(url)
    print("done")
