# coding: utf-8

"""日志设定文件"""

import logging

from longling.lib.stream import build_dir
from longling.lib.time import get_current_timestamp as default_timestamp

__all__ = ["LogLevel", "config_logging", "default_timestamp", "set_logging_info"]


class LogLevel(object):
    ERROR = logging.ERROR
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL


LogLevelDict = {}
for key, value in {
    "error": LogLevel.ERROR,
    "warn": LogLevel.WARN,
    "info": LogLevel.INFO,
    "debug": LogLevel.DEBUG,
    "critical": LogLevel.CRITICAL,
}.items():
    LogLevelDict[key] = value
    LogLevelDict[key.upper()] = value

LOG_COLORS = {
    'WARNING': 'y',
    'INFO': 'g',
    'DEBUG': 'b',
    'CRITICAL': 'y',
    'ERROR': 'r'
}


class _ColoredFormatter(logging.Formatter):
    """
    Formatter for colored log.
    Source code can be found in https://github.com/yxonic/fret/blob/master/fret/__init__.py
    """

    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = _colored(
                record.levelname,
                LOG_COLORS[record.levelname],
                style='b'
            )
        return logging.Formatter.format(self, record)


def _colored(fmt, fg=None, bg=None, style=None):
    """
    Return colored string.

    List of colours (for fg and bg):
        - k:   black
        - r:   red
        - g:   green
        - y:   yellow
        - b:   blue
        - m:   magenta
        - c:   cyan
        - w:   white

    List of styles:
        - b:   bold
        - i:   italic
        - u:   underline
        - s:   strike through
        - x:   blinking
        - r:   reverse
        - y:   fast blinking
        - f:   faint
        - h:   hide

    Parameters
    ----------
    fmt : str
        string to be colored
    fg : str
        foreground color
    bg : str
        background color
    style : str
        text style

    Examples
    --------
    >>> _colored("info")
    'info'
    >>> ass_str = r'\x1b[34;41minfo\x1b[0m'
    >>> ass_str == _colored("info", "b", "r")
    True
    """

    colcode = {
        'k': 0,  # black
        'r': 1,  # red
        'g': 2,  # green
        'y': 3,  # yellow
        'b': 4,  # blue
        'm': 5,  # magenta
        'c': 6,  # cyan
        'w': 7  # white
    }

    fmtcode = {
        'b': 1,  # bold
        'f': 2,  # faint
        'i': 3,  # italic
        'u': 4,  # underline
        'x': 5,  # blinking
        'y': 6,  # fast blinking
        'r': 7,  # reverse
        'h': 8,  # hide
        's': 9,  # strike through
    }

    # properties
    props = []
    if isinstance(style, str):
        props = [fmtcode[s] for s in style]
    if isinstance(fg, str):
        props.append(30 + colcode[fg])
    if isinstance(bg, str):
        props.append(40 + colcode[bg])

    # display
    props = ';'.join([str(x) for x in props])
    if props:
        return '\x1b[%sm%s\x1b[0m' % (props, fmt)
    else:
        return fmt


def config_logging(filename=None,
                   log_format=None,
                   level=logging.INFO,
                   logger=None, console_log_level=None, propagate=False,
                   mode='a',
                   file_format=None, encoding: (str, None) = "utf-8",
                   enable_colored=False, datefmt=None):
    """
    主日志设定文件

    Parameters
    ----------
    filename: str or None
        日志存储文件名，不为空时将创建文件存储日志
    log_format: str
        默认日志输出格式: %(name)s, %(levelname)s %(message)s
        如果 datefmt 被指定，则变为 %(name)s: %(asctime)s, %(levelname)s %(message)s
    level: str or int
        默认日志等级
    logger: str or logging.logger
        日志logger名，可以为空（使用root logger），
        字符串类型（创建对应名logger），logger
    console_log_level: str, int or None
        屏幕日志等级，不为空时，使能屏幕日志输出
    propagate: bool
    mode: str
    file_format: str or None
        文件日志输出格式，为空时，使用log_format
    encoding
    enable_colored: bool
    datefmt: str

    Returns
    -------

    """
    if logger is None:
        logger = logging.getLogger()
    elif isinstance(logger, str):
        logger = logging.getLogger(logger)

    if isinstance(level, str):
        level = LogLevelDict[level]
    if isinstance(console_log_level, str):
        console_log_level = LogLevelDict[console_log_level]

    # need a clean state, for some module may
    # have called logging functions already (i.e. logging.info)
    # in that case, a default handler would been appended,
    # causing undesired output to stderr
    for handler in logger.handlers:
        logger.removeHandler(handler)

    formatter_cls = logging.Formatter if enable_colored is False else _ColoredFormatter

    if datefmt is None:
        log_format = '%(name)s, %(levelname)s %(message)s' if log_format is None else log_format
    else:
        log_format = '%(name)s: %(asctime)s, %(levelname)s %(message)s' if log_format is None else log_format
        if datefmt == "default":
            datefmt = "%Y-%m-%d %H:%M:%S"

    formatter = formatter_cls(log_format, datefmt=datefmt)
    logger.setLevel(level)
    if not propagate:
        logger.propagate = False
    if filename and filename is not None:
        build_dir(filename)
        handler = logging.FileHandler(filename, mode=mode, encoding=encoding)
        file_formatter = formatter
        if file_format:
            file_formatter = formatter_cls(file_format, datefmt=datefmt)
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
    if console_log_level is not None:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter_cls(log_format, datefmt=datefmt))
        ch.setLevel(console_log_level)
        logger.addHandler(ch)

    return logger


def set_logging_info():
    logging.getLogger().setLevel(logging.INFO)
