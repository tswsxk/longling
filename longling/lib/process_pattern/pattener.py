import re

from .process_pattern_base import *

mode_dict = {}

__all__ = ["register", "mode_dict", "init_patterns", "line_init_patterns"]


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


def _init_line(line):
    try:
        line = line.strip()
        line1, line2 = line.split(':')
        return line, (line1, line2)
    except Exception:
        raise ValueError('error in %s' % line)


@register(0)
def _init_patterns0(line, pps=None):
    line, (line1, line2) = _init_line(line)

    try:
        v = line1.strip()
        ps1 = []
        p = re.compile(v)
        ps1.append(p)

        v = line2.strip()
        assert v

        ps2 = []
        p = re.compile(v)
        ps2.append(p)
    except Exception:  # pragma: no cover
        raise ValueError('error in %s' % line)

    if pps is not None and isinstance(pps, dict):
        pps[line] = (ps1, ps2)
    return line, (ps1, ps2)


@register(1)
def _init_patterns1(line, pps):
    line, (line1, line2) = _init_line(line)

    try:
        vs = line1.split()
        ps1 = []
        for v in vs:
            p = re.compile(v)
            ps1.append(p)

        vs = line2.split()
        ps2 = []
        for v in vs:
            p = re.compile(v)
            ps2.append(p)

    except Exception:  # pragma: no cover
        raise ValueError('error in %s' % line)

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
            except ValueError as e:  # pragma: no cover
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
        raise KeyError(
            "available mode is %s" % list(mode_dict.keys())
        )
    return mode_dict[mode](line, pps)
