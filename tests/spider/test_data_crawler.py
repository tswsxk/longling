# coding: utf-8
# create by tongshiwei on 2019/7/2

import pytest
from longling.spider.data_crawler import download_data


def test_download_data(tmp_path):
    url = "https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010"
    download_data(url, tmp_path, override=True)
    with pytest.raises(FileExistsError):
        download_data(url, tmp_path, override=False)
    download_data(url, tmp_path, override=True)
