# coding: utf-8
# 2020/3/10 @ tongshiwei

import functools
import time
import random
import json
from longling.lib.stream import as_io
from longling.lib.path import path_append, abs_current_dir
from longling.lib.candylib import as_list
from urllib.request import ProxyHandler, build_opener
from longling.spider.conf import logger

META_DATA = path_append(abs_current_dir(__file__), '../meta_data/')

HTTP_HEADER = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
    'Connection': 'keep-alive',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
}

with as_io(path_append(META_DATA, 'user_agents.txt')) as f:
    USER_AGENTS = [line.strip() for line in f.readlines()]

with as_io(path_append(META_DATA, 'IPpool.json')) as f:
    PROXIES = ["%s:%s" % (ip_port["ip"], ip_port["port"]) for ip_port in json.load(f)]


def get_proxies():
    return PROXIES


def get_http_header():
    # HTTP_HEADER.update(get_user_agent())
    return get_user_agent()


def get_opener():
    proxy_handler = ProxyHandler({
        'http': random.choice(get_proxies()),
    })
    opener = build_opener(proxy_handler)
    return opener


def get_user_agent():
    return {'User-Agent': random.choice(USER_AGENTS)}


def retry(max_retry=5, retry_interval=1, retry_errors=None, failed_exception=ConnectionError(), logger=logger):
    retry_errors = ConnectionResetError if retry_errors is None else retry_errors
    retry_errors = tuple(as_list(retry_errors))

    def retry_wrapper(f):
        @functools.wraps(f)
        def new_f(*args, **kwargs):
            for _ in range(max_retry):
                try:
                    return f(*args, **kwargs)
                except retry_errors as e:
                    time.sleep(retry_interval)
                    logger.debug(e)
            raise failed_exception

        return new_f

    return retry_wrapper
