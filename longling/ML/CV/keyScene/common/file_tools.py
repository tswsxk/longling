#!/usr/bin/env python
# encoding: utf-8

import os
import shutil


def get_all_file(dir_path, extensions=[]):
    '''
    @summary:获取文件夹下所有指定扩展类型的文件路径
    '''
    dir_list = [dir_path]
    file_list = []
    while len(dir_list) != 0:
        curr_dir = dir_list.pop(0)
        for path_name in os.listdir(curr_dir):
            full_path = os.path.join(curr_dir, path_name)
            if os.path.isdir(full_path):
                dir_list.append(full_path)
            else:
                extension = os.path.splitext(full_path)[1][1:]
                if len(extensions) == 0 or extension in extensions:
                    file_list.append(full_path)
    return file_list

def ensure_dir_exists(dir_name):
    '''
    @summary:判断文件夹是否存在，如不存在则创建文件夹
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    '''
    @summary:判断文件夹是否存在如果存在则删除文件夹
    '''
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def remove_file(file_name):
    '''
    @summary:判断文件是否存在，如果存在则删除文件
    '''
    if os.path.exists(file_name):
        os.remove(file_name)

