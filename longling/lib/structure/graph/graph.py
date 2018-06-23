# coding: utf-8
# create by tongshiwei on 2018/6/23


class EmptyEdge(object):
    pass


class Graph(object):
    def __init__(self, vertex_class, edge_class=tuple, vertexes_class=set, edges_class=set):
        self.vertexes = vertexes_class()
        self.edges = edges_class()
        self.vertex_class = vertex_class
        self.edge_class = edge_class

    def add_vertex(self, *node_info):
        self.vertexes.add(self.vertex_class(*node_info))

    def add_edge(self, *edge_info):
        if type(self.edge_class) is EmptyEdge:
            pass
        edge = self.edge_class(edge_info)
        self.edges.add(edge)

    def link(self, node_1, node_2, edge_info=None):
        raise NotImplementedError


class UndirectedGraph(Graph):
    def link(self, node_1, node_2, edge_info=None):
        node_1.linkto(node_2)
        node_2.linkto(node_1)
        self.add_edge(node_1, node_2, edge_info)


class DirectedGraph(Graph):
    def link(self, precursor, successor, edge_info=None):
        precursor.linkto(successor)
        self.add_edge(precursor, successor, edge_info)

    def doubly_link(self, node_1, node_2, edge_info=None):
        self.link(node_1, node_2, edge_info)
        self.link(node_2, node_1, edge_info)
