# coding: utf-8
# 2019/9/20 @ tongshiwei

r"""
To replace the variable str which is in the pattern r"(\$[A-z_]*)\b" （i.e., `$xxx`)
"""

from typing import overload

__all__ = ["variable_replace", "default_variable_replace"]

import re

VARIABLE_REGEX = r"(\$[A-z_]*)"
VARIABLE = re.compile(VARIABLE_REGEX)


def variable_replace(string: str, key_lower: bool = True, quotation: str = "", **variables):
    r"""
    Examples
    --------
    >>> string = "hello $who"
    >>> variable_replace(string, who="world")
    'hello world'
    >>> string = "hello $WHO"
    >>> variable_replace(string, key_lower=False, WHO="world")
    'hello world'
    >>> string = "hello $WHO"
    >>> variable_replace(string, who="longling")
    'hello longling'
    >>> string = "hello $Wh_o"
    >>> variable_replace(string, wh_o="longling")
    'hello longling'
    """

    def replace(match):
        variable = string[match.start() + 1:match.end()]
        variable = variable.lower() if key_lower else variable
        variable = variables[variable]
        return quotation + variable + quotation

    return VARIABLE.sub(replace, string)


@overload
def default_variable_replace(string: str, default_value: dict, key_lower: bool = True, quotation: str = "",
                             **variables) -> str:  # pragma: no cover
    # coverage cannot test this function
    """
    Examples
    --------
    >>> string = "hello $who, I am $author"
    >>> default_variable_replace(string, default_value={"author": "groot"}, who="world")
    'hello world, I am groot'
    >>> string = "hello $who, I am $author"
    >>> default_variable_replace(string, default_value={"author": "groot"})
    'hello , I am groot'
    """

    def replace(match):
        variable = string[match.start() + 1:match.end()]
        variable = variable.lower() if key_lower else variable
        if variable not in variables:
            return default_value.get(variable, "")
        return quotation + str(variables[variable]) + quotation

    return VARIABLE.sub(replace, string)


def default_variable_replace(string: str, default_value: (str, None) = "", key_lower: bool = True, quotation: str = "",
                             **variables) -> str:
    """
    Examples
    --------
    >>> string = "hello $who, I am $author"
    >>> default_variable_replace(string, who="world")
    'hello world, I am '
    >>> string = "hello $who, I am $author"
    >>> default_variable_replace(string, default_value=None, who="world")
    'hello world, I am $author'
    """

    def replace(match):
        variable = string[match.start() + 1:match.end()]
        variable = variable.lower() if key_lower else variable
        if variable not in variables:
            if default_value is not None:
                return default_value
            else:
                variable = string[match.start():match.end()]
                variable = variable.lower() if key_lower else variable
                return variable
        return quotation + str(variables[variable]) + quotation

    return VARIABLE.sub(replace, string)
