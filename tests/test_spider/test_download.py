# coding: utf-8
# create by tongshiwei on 2019/7/2

import pytest
from longling import path_append
from longling.spider.download_data import download_file
from longling.spider.utils import reporthook4urlretrieve
from longling.lib.path import file_exist


def test_download_data(tmp_path):
    url = "http://base.ustc.edu.cn/data/ASSISTment/2015_100_skill_builders_main_problems.zip "
    tmp_file = path_append(tmp_path, "download", to_str=True)

    download_file(url, decomp=False)

    download_file(url)

    download_file(url, tmp_file)
    download_file(url, tmp_file, reporthook=reporthook4urlretrieve)

    with pytest.raises(FileExistsError):
        download_file(url, tmp_file, override=False)
