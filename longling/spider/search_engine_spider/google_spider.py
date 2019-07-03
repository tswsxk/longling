# coding: utf-8

import requests
from bs4 import BeautifulSoup
import re
import json

def keyword_search(keyword, pages=1, fast_tag=False):
    ress = keyword_search_pages(keyword, pages)
    for res in ress:
        for r in get_title_and_url(res.text, fast_tag=fast_tag):
            yield r

def keyword_search_page(keyword, page_number=0):
    res = requests.get("https://www.google.com.hk/search?q=%s&start=%s" % (keyword, page_number * 10))
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

    links = soup.select("cite")
    titles = soup.select('div h3 a')

    for title, link in zip(titles, links):
        url = link.text
        if u'http' not in url:
            url = u"https://" + url
        if not fast_tag:
            try:
                res = requests.get(url, allow_redirects=False)
            except:
                continue
            if res.status_code == 200:
                try:
                    soup = BeautifulSoup(res.text, 'lxml')
                    urls = soup.select('head > noscript')
                    url2 = urls[0]
                    url_math = re.search(r'\'(.*?)\'', str(url2), re.S)
                    web_url = url_math.group(1)
                except:
                    web_url = url
            elif res.status_code == 302:
                web_url = res.headers['location']
                if u'http' not in web_url:
                    continue
            else:
                continue
        else:
            web_url = url

        yield {
            'title': title.get_text(),
            'url': web_url,
        }

if __name__ == '__main__':
    for data in keyword_search("如何调出日式小清新", 10):
        print json.dumps(data, ensure_ascii=False)

    # res = keyword_search_page("如何调出日式小清新")
    # soup = BeautifulSoup(res.text, "lxml")
    # print soup.prettify()
