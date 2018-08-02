# coding: utf-8
# create by tongshiwei on 2018/6/23
'''
advice using set structure to record the neighbors, precursor or successor
'''

'''
注意： 连边时不检查是否存在重复
'''
from longling.lib.structure.base import AddList
from longling.lib.candylib import get_all_subclass


class GraphNode(object):
    def __init__(self, value, *args, **kwargs):
        self.value = value
        self._id = None
        self.set_id(kwargs.get('id', None))

    def set_id(self, _id):
        self._id = _id

    @property
    def id(self):
        return self._id

    def __repr__(self):
        return str(self._id)


class UndirectedGraphNode(GraphNode):
    def __init__(self, value, neighbors=None, *args, **kwargs):
        super(UndirectedGraphNode, self).__init__(value, *args, **kwargs)
        self.neighbors = neighbors if neighbors else AddList()

    def add_neighbor(self, new_neighbor):
        self.neighbors.add(new_neighbor)

    def linkto(self, node):
        self.add_neighbor(node)

    @property
    def degree(self):
        return len(self.neighbors)


class DirectedGraphNode(GraphNode):
    def __init__(self, value, precursor=None, successor=None, *args, **kwargs):
        super(DirectedGraphNode, self).__init__(value, *args, **kwargs)
        self.precursor = precursor if precursor else AddList()
        self.successor = successor if successor else AddList()

    @property
    def neighbors(self):
        return self.precursor + self.successor

    def linkto(self, node):
        self.successor.add(node)
        node.precursor.add(self)

    @property
    def in_degree(self):
        return len(self.precursor)

    @property
    def out_degree(self):
        return len(self.successor)


GraphNodeType = tuple(get_all_subclass(GraphNode))

if __name__ == '__main__':
    print(GraphNodeType)
