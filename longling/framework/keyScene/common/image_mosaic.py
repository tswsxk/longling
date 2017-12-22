#!/usr/bin/env python
# coding=utf-8
import time
import os
from pyutil.image.mosaic import MosaicClient
from videoarch.tasks.common.metrics_tools import MetricsEmitter
from videoarch.tasks.common import is_i18n_env, get_conf

global _offline_mosaic_client
_offline_mosaic_client = None
def get_offline_mosaic_client():
    global _offline_mosaic_client
    if not _offline_mosaic_client:
        VIDEO_OFFLINE_NAMESPACE = 'video-cover-offline'
        VIDEO_OFFLINE_TOKEN = 'de88a93d1cc496df6e50f83f55ca2270'
        _offline_mosaic_client = MosaicClient(VIDEO_OFFLINE_TOKEN, namespace=VIDEO_OFFLINE_NAMESPACE)
    return _offline_mosaic_client

global _mosaic_client
_mosaic_client = None
def get_mosaic_client():
    global _mosaic_client
    if not _mosaic_client:
        mosaic_namespace = 'video-cover'
        mosaic_token = '1099f8007f24e04e0da56c371da28324'
        _mosaic_client = MosaicClient(mosaic_token, namespace=mosaic_namespace)
    return _mosaic_client


# ------store image------
def put_image_to_mosaic(filename):
    filesize = os.path.getsize(filename)
    if filesize == 0:
        raise Exception("put file size is 0")
    start_time = time.time()
    image_uri = _put_image_to_mosaic(filename)
    end_time = time.time()
    MetricsEmitter.emit_timer("image.put(mb/s)", filesize/((end_time-start_time)*1024*1024))
    return image_uri

def _put_image_to_mosaic(filename):
    if is_i18n_env():
        return put_image_to_image_service(filename)
    with open(filename) as image_file:
        file_data = image_file.read()
        infos = get_mosaic_client().post_image(file_data)
        image_uri = infos.get('key', '')
        return image_uri

def put_raw_image_to_mosaic(filename):
    filesize = os.path.getsize(filename)
    if filesize == 0:
        raise Exception("put file size is 0")
    start_time = time.time()
    image_uri = _put_raw_image_to_mosaic(filename)
    end_time = time.time()
    MetricsEmitter.emit_timer("image.raw.put(mb/s)", filesize/((end_time-start_time)*1024*1024))
    return image_uri

VIDEO_NAMESPCE = '1099f8007f24e04e0da56c371da28324'
def _put_raw_image_to_mosaic(filename, namespace=VIDEO_NAMESPCE):
    if is_i18n_env():
        return put_image_to_image_service(filename)
    with open(filename) as image_file:
        file_data = image_file.read()
        clinet = MosaicClient(namespace)
        res = clinet.post_raw(file_data, "image/webp")
        image_uri = res.get("key", "")
        return image_uri

#################
# for i18n
# service owner: jiaojian, zhouyongjia
#################
global _image_service_client
_image_service_client = None
def _get_image_service_client():
    global _image_service_client
    if _image_service_client:
        return _image_service_client

    from ss_thrift_gen.image_service import ImageService
    from pyutil.thrift.thrift_client import ThriftClient
    _host = get_conf().get_values('imagestore_host')
    _port = get_conf().get_values('imagestore_port')
    _image_service_client = ThriftClient(ImageService, _host, _port)
    return _image_service_client

global _image_putopt
_image_putopt = None
def _get_putopt():
    global _image_putopt
    if _image_putopt:
        return _image_putopt
    from ss_thrift_gen.image_service.ttypes import PutOpt
    _image_putopt = PutOpt(False)
    return _image_putopt

def put_image_to_image_service(filename):
    with open(filename) as image_file:
        file_data = image_file.read()
        _rsp = _get_image_service_client().PutOnline(file_data, _get_putopt())
        if _rsp.status != 'OK':
            raise Exception("put_image_to_image_service %s" % _rsp)
        return _rsp.key

if __name__ == "__main__":
    image_uri = put_image_to_mosaic('test.jpg')
