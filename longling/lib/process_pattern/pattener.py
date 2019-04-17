from __future__ import absolute_import

import re

from .process_pattern_base import *

mode_dict = {}


def register(number):
    def _register(func):
        if number in mode_dict:
            logger.warning(
                "mode-%s %s existed, overriding by %s"
                % (number, mode_dict[number].__name__, func.__name__)
            )
        else:
            logger.info("register mode-%s %s" % (number, func.__name__))
        mode_dict[number] = func
        return func

    return _register


@register(0)
def _init_patterns0(line, pps=None):
    try:
        line = line.decode('utf-8').strip()
    except AttributeError:
        pass
    try:
        line1, line2 = line.split(':')
    except Exception:
        raise ProcessPatternLineError('error in %s' % line)

    v = line1.strip()
    ps1 = []
    if v:
        try:
            p = re.compile(v)
        except Exception:
            raise ProcessPatternLineError('error in %s' % line)
        ps1.append(p)

    v = line2.strip()
    assert v
    ps2 = []
    try:
        p = re.compile(v)
    except Exception:
        raise ProcessPatternLineError('error in %s' % line)
    ps2.append(p)

    if pps is not None and isinstance(pps, dict):
        pps[line] = (ps1, ps2)
    return line, (ps1, ps2)


@register(1)
def _init_patterns1(line, pps):
    try:
        line = line.decode('utf-8').strip()
    except AttributeError:
        pass
    try:
        line1, line2 = line.split(':')
    except Exception:
        raise ProcessPatternLineError('error in %s' % line)

    vs = line1.split()
    ps1 = []
    for v in vs:
        try:
            p = re.compile(v)
        except Exception:
            raise ProcessPatternLineError('error in %s' % line)
        ps1.append(p)

    vs = line2.split()
    ps2 = []
    for v in vs:
        try:
            p = re.compile(v)
        except Exception:
            print("error:", v)
            raise ProcessPatternLineError('error in %s' % line)
        ps2.append(p)

    if pps is not None and isinstance(pps, dict):
        pps[line] = (ps1, ps2)

    return line, (ps1, ps2)


def _init_patterns(location, patterns):
    """
    创建一个模式字典

    Parameters
    ----------
    location: str
        正则规则文件位置
    patterns:

    Returns
    -------
    dict
        模式字典
    """
    pps = {}
    with open(location) as f:
        for line in f:
            try:
                patterns(line, pps)
            except ProcessPatternLineError as e:
                print(e)
                continue
    return pps


def init_patterns(location, mode):
    return _init_patterns(location, mode_dict[mode])


def line_init_patterns(line, mode, pps=None):
    """

    Parameters
    ----------
    line
    mode: int
    pps

    Returns
    -------

    """
    if mode not in mode_dict:
        raise ProcessPatternNotExistedPatternError(
            "available mode is %s" % list(mode_dict.keys())
        )
    return mode_dict[mode](line, pps)
