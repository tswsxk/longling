# coding: utf-8
# create by tongshiwei on 2019/7/2

import pytest
from longling import path_append
from longling.spider.download_data import download_file
from longling.spider.utils import reporthook4urlretrieve


def test_download_data(tmp_path):
    url = "http://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip"
    tmp_file = path_append(tmp_path, "download", to_str=True)

    download_file(url, save_path=tmp_file, decomp=False)

    download_file(url, tmp_file)
    download_file(url, tmp_file, reporthook=reporthook4urlretrieve)

    with pytest.raises(FileExistsError):
        download_file(url, tmp_file, override=False)
