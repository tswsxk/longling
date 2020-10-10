# coding: utf-8
# 2020/8/23 @ tongshiwei


from collections import Iterable
import bisect

__all__ = ["SortedList"]


class SortedList(list):
    """
    A list maintaining the element in an ascending order.

    A custom key function can be supplied to customize the sort order.


    Examples
    --------
    >>> sl = SortedList()
    >>> sl.adds(*[1, 2, 3, 4, 5])
    >>> sl
    [1, 2, 3, 4, 5]
    >>> sl.add(7)
    >>> sl
    [1, 2, 3, 4, 5, 7]
    >>> sl.add(6)
    >>> sl
    [1, 2, 3, 4, 5, 6, 7]
    >>> sl = SortedList([4])
    >>> sl.add(3)
    >>> sl.add(2)
    >>> sl
    [2, 3, 4]
    >>> list(reversed(sl))
    [4, 3, 2]
    >>> sl = SortedList([("harry", 1), ("tom", 0)], key=lambda x: x[1])
    >>> sl
    [('tom', 0), ('harry', 1)]
    >>> sl.add(("jack", -1), key=lambda x: x[1])
    >>> sl
    [('jack', -1), ('tom', 0), ('harry', 1)]
    >>> sl.add(("ada", 2))
    >>> sl
    [('jack', -1), ('tom', 0), ('harry', 1), ('ada', 2)]
    """

    def __init__(self, iterable: Iterable = (), key=None):
        super(SortedList, self).__init__(iterable)
        self._key = key
        self.sort(key=self._key)

    def _get_key(self, elem, key=None):
        if key is not None:
            return key(elem)
        elif self._key is not None:
            return self._key(elem)
        else:
            return elem

    def add(self, elem, key=None):
        if not self:
            self.append(elem)
        elif self._get_key(elem, key) >= self._get_key(self[-1], key):
            self.append(elem)
        elif self._get_key(elem, key) <= self._get_key(self[0], key):
            self.insert(0, elem)
        else:
            idx = bisect.bisect(self, elem)
            self.insert(idx, elem)

    def adds(self, *elem):
        for _elem in elem:
            self.add(_elem)
