# coding:utf-8
# created by tongshiwei on 2018/7/26

from gluonnlp.data import ConstWidthBucket, ExpWidthBucket, LinearWidthBucket


def test_example():
    """
    >>> schemer = ConstWidthBucket()
    >>> print(schemer(100, 10, 5))
    [100, 82, 64, 46, 28]
    >>> schemer = ExpWidthBucket()
    >>> print(schemer(100, 10, 5))
    [25, 41, 59, 78, 100]
    >>> schemer = LinearWidthBucket()
    >>> print(schemer(100, 10, 5))
    [17, 29, 47, 71, 100]
    """
    pass


if __name__ == '__main__':
    test_example()

    import doctest

    doctest.testmod()
