# coding: utf-8
# 2019/9/21 @ tongshiwei


def legal_input(__promt: str,
                __legal_input: set = None, __illegal_input: set = None, is_legal=None,
                __default_value: str = None):
    """
    To make sure the input legal, if the input is illegal, the user will be asked to retype the input.

    Input being illegal means the input either not in __legal_input or in __illegal_input.

    For the case that user do not type in anything, if the __default_value is set, the __default_value will be used.
    Else, the input will be treated as illegal.

    Parameters
    ----------
    __promt: str
        tips to be displayed
    __legal_input: set
        the input should be in the __legal_input set
    __illegal_input: set
        the input should be not in the __illegal_input set
    is_legal: function
        the function used to judge whether the input is legal, by default, we use the inner __is_legal function
    __default_value: str
        default value when user type in nothing

    """
    inp = input(__promt)
    __legal_input = __legal_input if __legal_input else set()
    __illegal_input = __illegal_input if __illegal_input else set()

    def __is_legal(__inp):
        if (not __legal_input or __inp in __legal_input) and __inp not in __illegal_input:
            # whether the input either not in __legal_input or in __illegal_input
            return True
        elif not inp and __default_value:
            # If the __default_value is set, the __default_value will be used.
            # Else, the input will be treated as illegal.
            return True
        return False

    is_legal = is_legal if is_legal else __is_legal

    while not is_legal(inp):
        # the user will be asked to retype the input.
        inp = input(__promt)
    if not inp and __default_value:
        return __default_value
    return inp


def default_legal_input(tip, __legal_input: set = None, __illegal_input: set = None, is_legal=None,
                        __default_value: str = None, cursor="<"):
    __promt = tip

    legal_tip = []

    if __legal_input is not None:
        legal_tip.append("/".join(__legal_input))

    if __default_value is not None:
        legal_tip.append("default is %s" % __default_value)

    if legal_tip:
        __promt = tip + " (%s)" % ", ".join(legal_tip)

    __promt += " %s " % cursor
    return legal_input(__promt, __legal_input=__legal_input, __illegal_input=__illegal_input, is_legal=is_legal,
                       __default_value=__default_value)


def binary_legal_input(tip, _y="y", _n="n", _default="y"):
    assert _default in [_y, _n, None]
    return True if legal_input("%s (%s/%s, default is %s) < " % (tip, _y, _n, _default),
                               __default_value=_default) == _y else False
