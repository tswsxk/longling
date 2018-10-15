# coding:utf-8
# created by tongshiwei on 2018/10/15

"""
XGBoost模板
"""
import xgboost as xgb


def get_data_iter(params):
    pass


if __name__ == '__main__':
    # # set parameters
    try:
        # for python module
        from .parameters import Parameters
    except (ImportError, SystemError):
        # for python script
        from parameters import Parameters
