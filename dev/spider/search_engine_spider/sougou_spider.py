# coding: utf-8

import re

import requests
from bs4 import BeautifulSoup

from longling.framework import headers, cookies


def keyword_search(keyword, pages=1):
    ress = keyword_search_pages(keyword, pages)
    for res in ress:
        for r in get_title_and_url(res.text):
            yield r

def keyword_search_page(keyword, page_number=0):
    res = requests.get("https://www.sogou.com/web?query=%s&page=%s" % (keyword, page_number + 1), cookies=cookies, headers=headers)
    return res

def keyword_search_pages(keyword, pages=1):
    for i in range(pages):
        yield keyword_search_page(keyword, i)

def get_title(resquest_data_text):
    soup = BeautifulSoup(resquest_data_text, 'lxml')
    titles = soup.select('div > h3 > a')
    return titles

def get_title_and_url(request_data_text, wf=None, fast_tag=False):
    soup = BeautifulSoup(request_data_text, 'lxml')

    titles = soup.select('div > h3 > a')

    for title, link in zip(titles, titles):

        baidu_url = link.get('href')
        if not fast_tag and str(baidu_url).find('link?url=') > 0:
            res = requests.get(baidu_url, allow_redirects=False)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'lxml')
                urls = soup.select('head > noscript')
                url2 = urls[0]
                url_math = re.search(r'\'(.*?)\'', str(url2), re.S)
                web_url = url_math.group(1)
            elif res.status_code == 302:
                web_url = res.headers['location']
            else:
                web_url = 'error'
        else:
            web_url = baidu_url

        yield {
            'title': title.get_text(),
            'url': web_url.encode('utf-8'),
        }

if __name__ == '__main__':
    # for data in keyword_search("test", 10):
    #     print json.dumps(data, ensure_ascii=False)

    res = keyword_search_page("如何调出日式小清新")
    soup = BeautifulSoup(res.text, "lxml")
    print soup.prettify()
