# coding: utf-8
# create by tongshiwei on 2018/6/23
'''
advice using set structure to record the neighbors, precursor or successor
'''


class GraphNode(object):
    def __init__(self, value, *args, **kwargs):
        self.value = value


class UndirectedGraphNode(GraphNode):
    def __init__(self, value, neighbors=None, *args, **kwargs):
        super(UndirectedGraphNode, self).__init__(value)
        self.neighbors = neighbors

    def add_neighbor(self, new_neighbor):
        self.neighbors.add(new_neighbor)

    def linkto(self, node):
        self.add_neighbor(node)


class DirectedGraphNode(GraphNode):
    def __init__(self, value, precursor, successor, *args, **kwargs):
        super(DirectedGraphNode, self).__init__(value)
        self.precursor = precursor
        self.successor = successor

    @property
    def neighbors(self):
        return self.precursor + self.successor

    def linkto(self, node):
        self.successor.add(node)
