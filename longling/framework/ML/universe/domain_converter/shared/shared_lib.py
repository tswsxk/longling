# coding: utf-8
# created by tongshiwei on 17-11-8


from output_format import output_str_format, output_numpy_format


def join(res, data):
    if isinstance(data, list):
        res.extend(data)
    else:
        res.append(data)


def output_format(data, output_format_type='str', **kwargs):
    if output_format_type == 'str':
        return output_str_format(data, **kwargs)

    elif output_format_type == 'numpy':
        return output_numpy_format(data)

    else:
        raise Exception('unknown output_format_type')
