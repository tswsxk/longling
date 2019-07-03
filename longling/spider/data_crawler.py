# coding: utf-8
# create by tongshiwei on 2019/7/2

from longling import wf_open
from longling import path_append, file_exist

import requests


def download_data(url, data_dir, override=False):
    """

    Parameters
    ----------
    url: str
    data_dir: str or Path
    override: bool

    Returns
    -------
    local_filename
    """
    local_filename = path_append(data_dir, url.split('/')[-1], to_str=True)

    if file_exist(local_filename):
        if not override:
            raise FileExistsError

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with wf_open(local_filename, mode="wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename
