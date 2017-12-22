# coding: utf-8
import urllib

def url_download(durl, filename=None, dirname=""):
    if filename is None or not filename:
        filename = durl.split("/")[-1]
    filename = dirname + filename
    status = urllib.urlretrieve(durl, filename)
    return status

if __name__ == '__main__':
    print url_download("https://pic3.zhimg.com/daddeafad987ac0a44e546af76a3ca6a_xs.jpg")