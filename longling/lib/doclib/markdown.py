# coding: utf-8
# create by tongshiwei on 2018/8/22


def csv_string_to_table(long_string, delimiter=',', col_num=None):
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
    output = ""
    row_data = [list(map(lambda x: x.strip(), _.split(delimiter))) for _ in long_string.strip().split("\n") if _]
    if col_num is not None:
        row_data[0] = [''] * (col_num - len(row_data[0])) + row_data[0]
    else:
        col_num = len(row_data[0])
    output += '|' + '|'.join(row_data[0]) + '\n'
    print('|' + '|'.join(row_data[0]))
    table_delimiter_line = '|' + '|'.join(['---'] * col_num)
    output += table_delimiter_line + '\n'
    print(table_delimiter_line)
    for data in row_data[1:]:
        output += '|' + '|'.join(data) + '\n'
        print('|' + '|'.join(data))

    return output


if __name__ == '__main__':
    long_string = '''
         exercise_length  exercise_num    last_time
    count     50002.000000  50002.000000  50002.00000
    mean         27.673873      2.833487    386.93766
    std          42.860613      3.816037    518.76202
    min           1.000000      1.000000      0.00000
    10%           1.000000      1.000000      0.00000
    25%           4.000000      1.000000     48.95350
    50%          11.000000      1.000000    201.17450
    75%          33.000000      3.000000    518.81725
    90%          72.000000      6.000000   1024.00000
    max        1107.000000    143.000000   7573.38600
 '''
    csv_string_to_table(long_string, None, 4)
