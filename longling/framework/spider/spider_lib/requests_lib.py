import requests


def conf_request(url, cookies=source.framework.spider.conf.cookies, headers=source.framework.spider.conf.headers):
    return requests.get(url, cookies=cookies, headers=headers)

def dyn_ua_requests(urls, cookies=source.framework.spider.conf.cookies, ua_list="doc/agent_list"):
    import random
    with open(ua_list) as f:
        uas = [line.strip() for line in f if '#' not in line]

    for url in urls:
        ua = uas[random.randint(0, len(uas) - 1)]
        headers = {
            'User-Agent': ua,
        }
        yield requests.get(url, cookies=cookies, headers=headers)