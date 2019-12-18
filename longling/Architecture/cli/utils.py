# coding: utf-8
# 2019/9/21 @ tongshiwei


def legal_input(__promt: str,
                __legal_input: set = None, __illegal_input: set = None, is_legal=None,
                __default_value=None):
    inp = input(__promt)
    __legal_input = __legal_input if __legal_input else set()
    __illegal_input = __illegal_input if __illegal_input else set()

    def __is_legal(__inp):
        if (not __legal_input or __inp in __legal_input) and __inp not in __illegal_input:
            return True
        elif not inp and __default_value:
            return True
        return False

    is_legal = is_legal if is_legal else __is_legal

    while not is_legal(inp):
        inp = input(__promt)
    if not inp and __default_value:
        return __default_value
    return inp
