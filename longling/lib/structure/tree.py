# coding: utf-8
# create by tongshiwei on 2018/6/23


class Tree(object):
    def __init__(self, value, *args, **kwargs):
        self.value = value

    @property
    def children(self):
        raise NotImplementedError


class BinaryTree(Tree):
    def __init__(self, value=None, left_child=None, right_child=None):
        super(BinaryTree, self).__init__(value)
        self.left_child = left_child
        self.right_child = right_child

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    @property
    def children(self):
        return [self.left_child, self.right_child]


class MultiChildrenTree(Tree):
    def __init__(self, value=None, children=None):
        super(MultiChildrenTree, self).__init__(value)
        self.__children = children

    @property
    def children(self):
        return self.__children
