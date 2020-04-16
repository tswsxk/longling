# coding: utf-8
# 2019/12/9 @ tongshiwei
import os
import gzip
import shutil
import tarfile
import zipfile
import rarfile

from longling import flush_print
from longling.lib.candylib import format_byte_sizeof

__all__ = ["decompress", "get_path", "un_zip", "un_rar", "un_tar", "reporthook4urlretrieve"]


def decompress(file):  # pragma: no cover
    for z in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar", ".gz"]:
        if file.endswith(z):
            if z == ".zip":
                un_zip(file)
            elif z == ".rar":
                un_rar(file)
            elif z in {".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz"}:
                un_tar(file)
            elif z == ".gz":
                un_gzip(file)
            break


def get_path(file):  # pragma: no cover
    #  返回解压缩后的文件名
    for i in [".tar.gz", ".tar.bz2", ".tar.bz", ".tar.tgz", ".tar", ".tgz", ".zip", ".rar", ".gz"]:
        file = file.replace(i, "")
    return file


def un_zip(file):  # pragma: no cover
    zip_file = zipfile.ZipFile(file)
    uz_path = get_path(file)
    print(file + " is unzip to " + uz_path)
    for name in zip_file.namelist():
        zip_file.extract(name, uz_path)
    zip_file.close()


def un_rar(file):  # pragma: no cover
    rar_file = rarfile.RarFile(file)
    uz_path = get_path(file)
    print(file + " is unrar to " + uz_path)
    rar_file.extractall(uz_path)


def un_tar(file):  # pragma: no cover
    tar_file = tarfile.open(file)
    uz_path = get_path(file)
    print(file + " is untar to " + uz_path)
    tar_file.extractall(path=uz_path)


def un_gzip(file):  # pragma: no cover
    uz_file = get_path(file)
    with gzip.open(file, 'rb') as f_in:
        with open(uz_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file)


def reporthook4urlretrieve(blocknum, bs, size):
    """

    Parameters
    ----------
    blocknum:
        已经下载的数据块
    bs:
        数据块的大小
    size:
        远程文件的大小

    Returns
    -------

    """
    per = 100.0 * (blocknum * bs) / size
    if per > 100:
        per = 100
    flush_print(
        'Downloading %.2f%% : %s | %s' % (
            per,
            format_byte_sizeof(blocknum * bs),
            format_byte_sizeof(size)
        ))
