# coding: utf-8
# create by tongshiwei on 2018/8/22


def csv_string_to_table(long_string, delimiter=','):
    """

    Parameters
    ----------
    long_string: str
    delimiter: str or None

    Examples
    --------
    >>> long_string = '''
    ... col_name1,col_name2, col_name3
    ... 1, 2, 3
    ... '''
    """
    row_data = [list(map(lambda x: x.strip(), _.split(delimiter))) for _ in long_string.strip().split("\n") if _]
    print('|'.join(row_data[0]))
    table_delimiter_line = '|'.join(['---'] * len(row_data[0]))
    print(table_delimiter_line)
    for data in row_data[1:]:
        print('|'.join(data))


if __name__ == '__main__':
    long_string = '''
        12884	148691	1420714809324	ATTEMPT	RESULT	telling-time	time_terminology	time_terminology--analog_word	1420714806324	time_terminology--analog_word	INCORRECT	Choose_Exercise	NA	NA	NA	NA	time_terminology	telling-time	arithmetic	0	0
12884	148691	1420714810324	ATTEMPT	RESULT	telling-time	time_terminology	time_terminology--analog_word	1420714809324	time_terminology--analog_word	INCORRECT	Choose_Exercise	NA	NA	NA	NA	time_terminology	telling-time	arithmetic	0	0

        '''
    csv_string_to_table(long_string, None)