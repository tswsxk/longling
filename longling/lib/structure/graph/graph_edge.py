# coding: utf-8
# create by tongshiwei on 2018/6/24


from longling.lib.candylib import get_all_subclass


class GraphEdge(object):
    def __init__(self, node_1, node_2, edge_weight, edge_type=None, directed=False):
        self.node_1 = node_1
        self.node_2 = node_2
        self.edge_type = edge_type
        self.edge_weight = edge_weight
        self.directed = directed


class MultiRelationGraphEdge(GraphEdge):
    def __init__(self, node_1, node_2, edge_weight, edge_type, directed=False):
        assert edge_type
        super(MultiRelationGraphEdge, self).__init__(
            node_1, node_2, edge_weight, edge_type, directed
        )

GraphEdgeType = tuple(get_all_subclass(GraphEdge))

if __name__ == '__main__':
    print(GraphEdgeType)
