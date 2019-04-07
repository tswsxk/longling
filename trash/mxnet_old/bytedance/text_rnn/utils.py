# -*- coding: utf-8 -*-


def to_unicode(s):
    return s.decode('utf-8') if isinstance(s, str) else s
