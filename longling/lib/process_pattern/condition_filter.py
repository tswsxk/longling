import re


class CommentFilter(object):
    # 评论内容过滤器，过滤 只含表情，数字 以及 回复（含有 //@ ）评论
    def __init__(self):
        self.emoji_re = re.compile('['
                                   '\U0001F300-\U0001F64F'
                                   '\U0001F680-\U0001F6FF'
                                   '\u2600-\u2B55]+',
                                   re.UNICODE)

    def is_valid(self, data):
        if re.sub("^\d+$", "", self.emoji_re.sub("", data['text'])):
            if "//@" not in data['text']:
                return True
        return False
