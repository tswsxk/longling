import re


class CommentFilter(object):
    """
    评论内容过滤器，过滤 只含表情，数字 以及 回复（含有 //@ ）评论

    Examples
    --------
    >>> cf = CommentFilter()
    >>> cf.is_valid("🐶😂")
    False
    >>> cf.is_valid("//@groot: hello world")
    False
    >>> cf.is_valid("233")
    False
    >>> cf.is_valid("再给我两分钟")
    True
    """

    def __init__(self):
        self.emoji_re = re.compile('['
                                   '\U0001F300-\U0001F64F'
                                   '\U0001F680-\U0001F6FF'
                                   '\u2600-\u2B55]+',
                                   re.UNICODE)

    def is_valid(self, string: str):
        if re.sub("^\d+$", "", self.emoji_re.sub("", string)):
            if "//@" not in string:
                return True
        return False
