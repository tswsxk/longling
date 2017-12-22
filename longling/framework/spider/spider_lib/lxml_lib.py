# coding: utf-8
from lxml import html
import re

HTML_COMMENT_PTN = re.compile(r'{!-- .*? --}')

def conv2unicode(x):
    x = x if isinstance(x, unicode) else x.decode('utf8')
    return x

def extract(request_data):
    data = request_data
    try:
        data = data.text
    except:
        pass
    try:
        data = html.fromstring(data)
    except:
        pass

    return data

def get_all_text(request_data_text):
    request_data = conv2unicode(request_data_text)
    if not request_data:
        return ''
    request_data = request_data + '<p></p>'
    node = html.fromstring(request_data)
    # ts = [i.text for i in node.iter("p")]
    # ts = [i.strip() for i in node.itertext() ]
    ts = [HTML_COMMENT_PTN.sub('', i.strip()) for i in node.itertext()]
    return '\n'.join([i for i in ts if i !=''])

def get_all_url(request_data, dup_detector=None):
    node = extract(request_data)
    urls = node.xpath('//@href')
    urls = filter(lambda x: 'http' in x, urls)
    if dup_detector is not None:
        urls = filter(lambda x: x not in dup_detector, urls)
    return urls

def get_all_img(request_data):
    node = extract(request_data)
    imgs = node.xpath('//img[@src]')
    for img in imgs:
        attrib = img.attrib
        if 'http' in attrib['src']:
            yield attrib['src'], attrib.get('alt', '')

def get_all_video(request_data):
    pass

if __name__ == '__main__':
    from requests_lib import *
    import json
    res = conf_request("https://www.zhihu.com/question/21358581")
    for r in get_all_img(res):
        print json.dumps(r, ensure_ascii=False)
