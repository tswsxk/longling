# coding: utf-8
# 2021/8/3 @ tongshiwei

"""
Here are some frequently used regex expression for select in collect_params()
"""
# 包含所有参数
ALL_PARAMS = None

# 除外所有embedding层
BLOCK_EMBEDDING = "^(?!.*embedding)"
