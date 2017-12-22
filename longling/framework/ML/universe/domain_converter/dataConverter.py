# coding: utf-8
# created by tongshiwei on 17-11-8

import logging

from shared import join, output_format

# each dataConverter can have one or many fieldConverter
# when convert, dataConverter will call different fieldConverter to process data accroding to the name of field

class baseDataConverter(object):
    def __init__(self, field_converters={}):
        self.field_converters = field_converters

    def convert(self, data, field_names):
        ret = []
        for i, field_name in enumerate(field_names):
            if field_name in self.field_converters:
                res = self.field_converters[field_name].convert(data[i])
            else:
                try:
                    res = float(data[i])
                except:
                    logging.error("error in convert %s-%s, set -1" % (data, data[i]))
                    res = -1

            self.join(ret, res)

        return self.output_format(ret)

    def join(self, res, data):
        return join(res, data)

    def output_format(self, data, output_format_type='str', **kwargs):
        return output_format(data, output_format_type, **kwargs)
