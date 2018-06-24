# coding: utf-8
# create by tongshiwei on 2018/6/23

from longling.lib.structure.base import AddList


class Graph(object):
    def __init__(self, vertexes_class=AddList, edges_class=dict):
        assert hasattr(AddList, 'add')
        self.vertexes = vertexes_class()
        self.edges = edges_class()

    def add_vertex(self, vertex):
        self.vertexes.add(vertex)

    def add_edge(self, nodes, edge):
        assert type(nodes) is tuple and len(nodes) == 2
        self.edges[nodes] = edge

    def _link(self, node_1, node_2, edge_info):
        raise NotImplementedError


class UndirectedGraph(Graph):
    def _link(self, node_1, node_2, edge):
        node_1.linkto(node_2)
        node_2.linkto(node_1)
        self.add_edge((node_1, node_2), edge)


class DirectedGraph(Graph):
    def _link(self, precursor, successor, edge):
        precursor.linkto(successor)
        self.add_edge((precursor, successor), edge)

    def _doubly_link(self, node_1, node_2, edge):
        self._link(node_1, node_2, edge)
        self._link(node_2, node_1, edge)
