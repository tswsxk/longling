# /usr/bin/env python
# coding=utf8

import binascii
import csv
import json
import string
import StringIO
from collections import namedtuple


def hex2binary(hex):
    try:
        res = binascii.unhexlify(hex)
    except TypeError as e:
        res = 0
    return res


def binary2hex(binary):
    res = binascii.hexlify(binary)
    return res


def json2obj(data):
    try:
        obj = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        return obj
    except Exception as _:
        return None


def csv2html(csv_file, first_line_header=True):
    container = "<table border=1>%s</table>"
    table_string = ""
    row_template = """
    <tr %s>
        <td>
        %s
        </td>
    </tr>
    """
    header_style = " style='background-color: #c5dfde;' "
    content_style = ""
    reader = csv.reader(csv_file)
    for row in reader:
        if first_line_header is True:
            style = header_style
            first_line_header = False
        else:
            style = content_style
        table_string += row_template % (style, string.join(row, "</td><td>"))
    return container % table_string


def dict2html(info, first_line_header=True, field_names=[]):
    # info: dict or list
    if isinstance(info, dict):
        info = [info]

    f = StringIO.StringIO()
    field_names = field_names or [key for key in info[0]]
    writer = csv.DictWriter(f, field_names)
    if first_line_header:
        writer.writeheader()
    writer.writerows(info)
    f.seek(0)
    html_table = csv2html(f)
    return html_table
