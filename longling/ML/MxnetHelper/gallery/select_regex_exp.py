# coding: utf-8
# create by tongshiwei on 2019/4/13
"""
Here are some frequently used regex expression for select in collect_params()
"""
# 包含所有参数
all_params = None

# 除外所有embedding层
block_embedding = "^(?!.*embedding)"
