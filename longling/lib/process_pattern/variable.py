# coding: utf-8
# 2019/9/20 @ tongshiwei

from typing import overload

__all__ = ["variable_replace", "default_variable_replace"]

import re

VARIABLE_REGEX = r"(\$[A-z]*)\b"
VARIABLE = re.compile(VARIABLE_REGEX)


@overload
def default_variable_replace(string: str, default_value: dict, key_lower: bool = True, quotation: str = "",
                             **variables) -> str:
    def replace(match):
        variable = string[match.start() + 1:match.end()]
        variable = variable.lower() if key_lower else variable
        if variable not in variables:
            return default_value.get(variable, "")
        return quotation + str(variables[variable]) + quotation

    return VARIABLE.sub(replace, string)


def default_variable_replace(string: str, default_value: str = "", key_lower: bool = True, quotation: str = "",
                             **variables) -> str:
    def replace(match):
        variable = string[match.start() + 1:match.end()]
        variable = variable.lower() if key_lower else variable
        if variable not in variables:
            return default_value
        return quotation + str(variables[variable]) + quotation

    return VARIABLE.sub(replace, string)


def variable_replace(string: str, key_lower: bool = True, quotation: str = "", **variables):
    def replace(match):
        variable = str(variables[string[match.start() + 1:match.end()]])
        variable = variable.lower() if key_lower else variable
        return quotation + variable + quotation

    return VARIABLE.sub(replace, string)
