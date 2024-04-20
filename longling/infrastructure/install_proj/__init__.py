# coding: utf-8
# 2019/9/20 @ tongshiwei

from .py_proj import py_proj
from .docs_proj import docs_proj
from .web_proj import web_proj

project_types = {
    "python": py_proj,
    "docs": docs_proj,
    "web": web_proj,
}
