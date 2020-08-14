# coding: utf-8
# create by tongshiwei on 2019/7/2

import logging

import os

from urllib.request import urlretrieve
from .utils import decompress, reporthook4urlretrieve as _reporthook4urlretrieve

logger = logging.getLogger("spider")

__all__ = ["download_file"]


def download_file(url, save_path=None, override=True, decomp=True, reporthook=None):
    """
    cli alias: ``download``, download data from specified url

    Parameters
    ----------
    url
    save_path
    override
    decomp
    reporthook

    Returns
    -------

    """
    save_path = url.split('/')[-1] if not save_path else save_path
    if os.path.exists(save_path):
        if override is True:
            os.remove(save_path)
            logger.warning(save_path + ' will be overridden.')
        else:
            raise FileExistsError("%s existed, downloading abandoned" % save_path)

    logger.info(url + ' is saved as %s', save_path)
    if reporthook is None:
        urlretrieve(url, save_path, reporthook=_reporthook4urlretrieve)
        print()
    else:
        urlretrieve(url, save_path, reporthook=reporthook)
    if decomp:
        decompress(save_path)
