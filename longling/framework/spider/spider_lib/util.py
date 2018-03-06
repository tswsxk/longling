# coding: utf-8
# created by tongshiwei on 18-1-28

from __future__ import absolute_import
from __future__ import print_function

import json
import os
import re
import requests

from collections import OrderedDict

from bs4 import BeautifulSoup
from tqdm import tqdm

from longling.framework.spider.spider_lib import url_download

from longling.lib.stream import check_dir, check_file, wf_open, wf_close

from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.layout import *


def get_target_article():
    with open("/home/tongshiwei/PycharmProjects/longling/data/spider/spider_targets.html", "rb") as f:
        r = f.read()
    res = BeautifulSoup(r, "html.parser")
    article_dict = OrderedDict()
    gsc_a_trs = res.find_all("tr", class_="gsc_a_tr")
    for gsc_a_tr in gsc_a_trs[6:]:
        gsc_a_t = gsc_a_tr.find("td", class_="gsc_a_t")
        article_title = re.sub("[ \t\n\r)]+", " ", gsc_a_t.a.text)
        authors = set([name.strip().lower() for name in re.split(",|，", gsc_a_t.find("div").text)])
        gsc_a_c = gsc_a_tr.find("td", class_="gsc_a_c")
        cited_link = gsc_a_c.a['href']
        cited_num = gsc_a_c.text
        cited_num = int(cited_num) if cited_num else 0
        if int(cited_num) <= 0:
            continue
        article_dict[article_title] = {"authors": authors, "link": cited_link, "num": cited_num, "cite_articles": {}}

    return article_dict


def pdf_read(filename):
    res = ""
    # 打开一个pdf文件
    fp = open(filename, 'rb')
    # 创建一个PDF文档解析器对象
    parser = PDFParser(fp)
    # 创建一个PDF文档对象存储文档结构
    # 提供密码初始化，没有就不用传该参数
    # document = PDFDocument(parser, password)
    document = PDFDocument(parser)
    # 检查文件是否允许文本提取
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed
    # 创建一个PDF资源管理器对象来存储共享资源
    # caching = False不缓存
    rsrcmgr = PDFResourceManager(caching=False)
    # 创建一个PDF设备对象
    laparams = LAParams()
    # 创建一个PDF页面聚合对象
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # 创建一个PDF解析器对象
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # 处理文档当中的每个页面

    # doc.get_pages() 获取page列表
    # for i, page in enumerate(document.get_pages()):
    # PDFPage.create_pages(document) 获取page列表的另一种方式
    replace = re.compile(r'\s+');
    # 循环遍历列表，每次处理一个page的内容
    page_num = 0
    for page in PDFPage.create_pages(document):
        page_num += 1
        interpreter.process_page(page)
        # 接受该页面的LTPage对象
        layout = device.get_result()
        # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
        # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
        for x in layout:
            # 如果x是水平文本对象的话
            if (isinstance(x, LTTextBoxHorizontal)):
                text = re.sub(replace, ' ', x.get_text())
                if len(text) != 0:
                    res += text
    return res, page_num, document, interpreter, device


def replace_braket(line):
    line = re.sub(r'\[', '\[', line)
    line = re.sub(r'\]', '\]', line)
    return line


def pdf_check(filename, article):
    print("pdf checking")
    text, page_num, document, interpreter, device = pdf_read(filename)

    chireg = re.compile(u'[\u4e00-\u9fa5]+')
    if chireg.search(text):
        return True, "", "", page_num

    print("finding ref_num")
    match_obj = re.search(r'((\[(\d+)\])|(\d+))\D*%s' % article, re.sub("\d{4}", "", text), re.M | re.I)

    if match_obj is None:
        return False, "", "", page_num

    full_ref = match_obj.group()
    ref_num1 = match_obj.group(1)
    ref_num2 = match_obj.group(2)
    ref_num = ref_num2 if ref_num1 is None else ref_num1

    check_reg = re.compile(r".{,50}%s.{,50}" % replace_braket(ref_num))

    replace = re.compile(r'\s+')

    res_dict = {}
    pages = PDFPage.create_pages(document)
    print("finding ref happened")
    for page_no, page in enumerate(pages):
        interpreter.process_page(page)
        # 接受该页面的LTPage对象
        layout = device.get_result()
        # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
        # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
        for x in layout:
            # 如果x是水平文本对象的话
            if (isinstance(x, LTTextBoxHorizontal)):
                text = re.sub(replace, ' ', x.get_text())
                if len(text) != 0:
                    check_lines = check_reg.findall(text)
                    if check_lines:
                        res_dict[str(page_no)] = check_lines
    return False, full_ref, res_dict, page_num


def qi_spider():
    article_dict = get_target_article()
    skip_set = {}
    start = 2
    end = 6
    article_id = 0
    min_cited = 0
    name_file = wf_open(os.path.join("/home/tongshiwei/PycharmProjects/longling/data/spider/", "name_exclude.txt"), "w")
    print("from,ref,from_users,ref_users,join", file=name_file)
    for article_t, infos in article_dict.items():
        article_id += 1
        if article_id > end:
            break
        if article_id in skip_set or article_id < start or infos['num'] < min_cited:
            print("skip %s %s" % (article_id, article_t))
            continue
        print("%s %s" % (article_id, article_t))
        url = infos['link']
        authors = infos['authors']
        left_num = infos['num']
        cite_articles = infos['cite_articles']
        visit_num = 0
        page_num = (left_num + 9) // 10
        dirname = "/home/tongshiwei/PycharmProjects/longling/data/spider/" + article_t
        record_f = wf_open(os.path.join(dirname, article_t + ".csv"), "w")
        tag_name = [
            "No", "article_title", "filename", "pdf_link", "authors",
            "download_info", "ref", "page_num", "check_tag", "scholar_url", "chitag",
            "useful",
        ]
        print(",".join(tag_name), file=record_f)
        not_access_f = wf_open(os.path.join(dirname, "n_acc.csv"), "w")
        print("title,link", file=not_access_f)
        for page in range(page_num):
            url_tail = "&start=%d" % (page * 10) if page else ""
            durl = url + url_tail
            filename = article_t + "%s.html" % ("_%s" % page if page else "")
            check_dir(dirname)
            if not check_file(os.path.join(dirname, filename)):
                try:
                    url_download(durl, filename=filename, dirname=dirname)
                except TypeError:
                    pass
            with open(os.path.join(dirname, filename), "rb") as f:
                bytes = f.read()

            res = BeautifulSoup(bytes, "lxml")

            gs_r_gs_or_gs_scl_s = res.find_all("div", class_="gs_r gs_or gs_scl")
            for gs_r_gs_or_gs_scl in gs_r_gs_or_gs_scl_s:
                left_num -= 1
                visit_num += 1

                article_authors = gs_r_gs_or_gs_scl.find("div", class_="gs_a").text
                article_authors = re.sub("….*", "", article_authors)
                article_authors = re.sub("[ \t\n\r)]+", " ", article_authors)
                article_authors = re.split(",|，", article_authors.lower())
                article_authors = set(article_authors)

                article_title = re.sub("[ \t\n\r)]+", " ", gs_r_gs_or_gs_scl.find("h3", class_="gs_rt").text)
                if authors & article_authors:
                    print("%s,%s,%s,%s,%s" % (article_t, article_title, "|".join(authors), "|".join(article_authors),
                                              "|".join(authors & article_authors)), file=name_file)
                    continue
    wf_close(name_file)
    pdf_link = gs_r_gs_or_gs_scl.a['href']
    article_title = re.sub("[ \t\n\r)]+", " ", gs_r_gs_or_gs_scl.find("h3", class_="gs_rt").text)
    cite_articles[article_title] = {
        "No": str(visit_num),
        "article_title": article_title,
        "scholar_url": durl,
        'pdf_link': pdf_link,
        'authors': ",".join(article_authors),
        'download_info': "",
        'filename': "",
        'ref': "",
        "res": "",
        "page_num": "",
        "check_tag": False,
        "chitag": 'False',
        "useful": 0,
    }
    if pdf_link:
        try:
            dres = url_download(pdf_link, filename=article_title + ".pdf", dirname=dirname, file_type="pdf", retry=5)
            if dres['status_code'] >= 0:
                cite_articles[article_title]['filename'] = dres['filename']
        except TypeError as e:
            cite_articles['download_info'] = e
            print(pdf_link, e)
            print("%s,%s" % (article_t, pdf_link), file=not_access_f)
        except EOFError as e:
            cite_articles['download_info'] = e
            print("%s,%s" % (article_t, pdf_link), file=not_access_f)
            print(pdf_link, e)
    save_filename = cite_articles[article_title].get('filename', '')
    if save_filename:
        try:
            chitag, ref, ref_res, page_num = pdf_check(save_filename, article_t)

            if not chitag:
                if ref and ref_res and page_num:
                    cite_articles[article_title]['page_num'] = str(page_num)
                    if page_num > 4:
                        cite_articles[article_title]['ref'] = ref
                        cite_articles[article_title]['ref_res'] = json.dumps(ref_res, ensure_ascii=False)

                        wf = wf_open(save_filename + ".csv")
                        print("page,content", file=wf)
                        print("-1,%s" % ref, file=wf)
                        rcnt = 0
                        for page, page_res in ref_res.items():
                            for r in page_res:
                                rcnt += 1
                                print("%s,%s" % (page, r), file=wf)
                        wf_close(wf)
                        if rcnt > 1:
                            cite_articles[article_title]['useful'] = 1
                        cite_articles[article_title]['check_tag'] = True
                else:
                    cite_articles[article_title]['check_tag'] = True
            else:
                cite_articles[article_title]['chitag'] = 'True'
        except Exception:
            pass
    cite_articles[article_title]['check_tag'] = str(cite_articles[article_title]['check_tag'])
    cite_articles[article_title]['useful'] = str(cite_articles[article_title]['useful'])
    res_record = []
    for key in tag_name:
        res_record.append(cite_articles[article_title].get(key, ''))
    print(",".join(res_record), file=record_f)

    wf_close(record_f)


if __name__ == '__main__':
    # article_t = "Collaborative learning team formation: a cognitive modeling perspective"
    # infos = {
    #     'link': "https://scholar.google.com/scholar?oi=bibs&hl=zh-CN&cites=6567333064625429316",
    #     'authors': set("Y Liu, Q Liu, R Wu, E Chen, Y Su, Z Chen, G Hu".lower().split(",")),
    #     'num': 5,
    #     'cite_articles': {},
    # }
    # url = infos['link']
    # authors = infos['authors']
    # left_num = infos['num']
    # cite_articles = infos['cite_articles']
    # visit_num = 0
    # page_num = (left_num + 9) // 10
    # dirname = "/home/tongshiwei/PycharmProjects/longling/data/spider/" + article_t
    # record_f = wf_open(os.path.join(dirname, article_t + ".csv"), "w")
    # tag_name = [
    #     "No", "article_title", "filename", "pdf_link", "authors",
    #     "download_info", "ref", "page_num", "check_tag", "scholar_url", "chitag",
    #     "useful",
    # ]
    # print(",".join(tag_name), file=record_f)
    # not_access_f = wf_open(os.path.join(dirname, "n_acc.csv"), "w")
    # print("title,link", file=not_access_f)
    # for page in range(page_num):
    #     url_tail = "&start=%d" % (page * 10) if page else ""
    #     durl = url + url_tail
    #     filename = article_t + "%s.html" % ("_%s" % page if page else "")
    #     check_dir(dirname)
    #     if not check_file(os.path.join(dirname, filename)):
    #         try:
    #             url_download(durl, filename=filename, dirname=dirname)
    #         except TypeError:
    #             pass
    #     with open(os.path.join(dirname, filename), "rb") as f:
    #         bytes = f.read()
    #
    #     res = BeautifulSoup(bytes, "lxml")
    #
    #     gs_r_gs_or_gs_scl_s = res.find_all("div", class_="gs_r gs_or gs_scl")
    #     for gs_r_gs_or_gs_scl in gs_r_gs_or_gs_scl_s:
    #         left_num -= 1
    #         visit_num += 1
    #
    #         article_authors = gs_r_gs_or_gs_scl.find("div", class_="gs_a").text
    #         article_authors = re.sub("….*", "", article_authors)
    #         article_authors = re.sub("[ \t\n\r)]+", " ", article_authors)
    #         article_authors = re.split(",|，", article_authors.lower())
    #         article_authors = set(article_authors)
    #
    #         if authors & article_authors:
    #             continue
    #
    #         pdf_link = gs_r_gs_or_gs_scl.a['href']
    #         article_title = re.sub("[ \t\n\r)]+", " ", gs_r_gs_or_gs_scl.find("h3", class_="gs_rt").text)
    #         cite_articles[article_title] = {
    #             "No": str(visit_num),
    #             "article_title": article_title,
    #             "scholar_url": durl,
    #             'pdf_link': pdf_link,
    #             'authors': ",".join(article_authors),
    #             'download_info': "",
    #             'filename': "",
    #             'ref': "",
    #             "res": "",
    #             "page_num": "",
    #             "check_tag": False,
    #             "chitag": 'False',
    #             "useful": 0,
    #         }
    #         if pdf_link:
    #             try:
    #                 dres = url_download(pdf_link, filename=article_title + ".pdf", dirname=dirname, file_type="pdf", retry=5)
    #                 if dres['status_code'] >= 0:
    #                     cite_articles[article_title]['filename'] = dres['filename']
    #             except TypeError as e:
    #                 cite_articles['download_info'] = e
    #                 print(pdf_link, e)
    #                 print("%s,%s" % (article_t, pdf_link), file=not_access_f)
    #             except EOFError as e:
    #                 cite_articles['download_info'] = e
    #                 print("%s,%s" % (article_t, pdf_link), file=not_access_f)
    #                 print(pdf_link, e)
    #         save_filename = cite_articles[article_title].get('filename', '')
    #         if save_filename:
    #             try:
    #                 chitag, ref, ref_res, page_num = pdf_check(save_filename, article_t)
    #
    #                 if not chitag:
    #                     if ref and ref_res and page_num:
    #                         cite_articles[article_title]['page_num'] = str(page_num)
    #                         if page_num > 4:
    #                             cite_articles[article_title]['ref'] = ref
    #                             cite_articles[article_title]['ref_res'] = json.dumps(ref_res, ensure_ascii=False)
    #
    #                             wf = wf_open(save_filename + ".csv")
    #                             print("page,content", file=wf)
    #                             print("-1,%s" % ref, file=wf)
    #                             rcnt = 0
    #                             for page, page_res in ref_res.items():
    #                                 for r in page_res:
    #                                     rcnt += 1
    #                                     print("%s,%s" % (page, r), file=wf)
    #                             wf_close(wf)
    #                             if rcnt > 1:
    #                                 cite_articles[article_title]['useful'] = 1
    #                             cite_articles[article_title]['check_tag'] = True
    #                     else:
    #                         cite_articles[article_title]['check_tag'] = True
    #                 else:
    #                     cite_articles[article_title]['chitag'] = 'True'
    #             except Exception:
    #                 pass
    #         cite_articles[article_title]['check_tag'] = str(cite_articles[article_title]['check_tag'])
    #         cite_articles[article_title]['useful'] = str(cite_articles[article_title]['useful'])
    #         res_record = []
    #         for key in tag_name:
    #             res_record.append(cite_articles[article_title].get(key, ''))
    #         print(",".join(res_record), file=record_f)
    #
    # wf_close(record_f)
    qi_spider()
