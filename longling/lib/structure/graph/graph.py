# coding: utf-8
# create by tongshiwei on 2018/6/23

from longling.lib.structure.base import AddList, logger
from longling.lib.structure.graph.graph_edge import GraphEdgeType
from longling.lib.structure.graph.graph_node import GraphNodeType


def get_id(node):
    return node.id if hasattr(node, 'id') and node.id is not None else str(node)


def string(node):
    return str(node) if node is not None else None


class GraphRelationType(object):
    SingleRelation = 0
    MultiRelation = 1


class GraphRelationTypeError(TypeError):
    pass


class GraphPlotTypeError(TypeError):
    pass


EDGE_CLASS = (None, int, float, tuple, list) + GraphEdgeType
VERTEX_CLASS = (None,) + GraphNodeType


class Graph(object):
    def __init__(self, vertexes_class=AddList, edges_class=dict, vertex_class=None, edge_class=None, logger=logger):
        '''

        :param vertexes_class:
        :param edges_class:
        :param vertex_class:
        :param edge_class: when edge class is None, it is a simple graph without weight and relation type;
        when edge class in (int float), the graph is a simple weighted graph without relation type; in other conditions,
        when relation_type is specified, the graph will become multi-relational graph if there are several relation,
        which can be refered by querying attribute 'graph_relation_type'. In visualization, the relation will be first
        prior to be shown while the weight next. More information to be shown in graph can be specified in the 'info'
        attribute of the edge_class.
        :param logger:
        '''
        assert hasattr(vertexes_class, 'add'), 'vertex_class must have add attribute'
        assert edge_class in EDGE_CLASS, 'edge class must be in %s, now is %s' % (EDGE_CLASS, edge_class)
        assert vertex_class in VERTEX_CLASS, 'vertex class must be in %s, now is %s' % (VERTEX_CLASS, vertex_class)
        self.vertexes = vertexes_class()
        self.edges = edges_class()
        self.vertex_class = vertex_class
        self.edge_class = edge_class
        self.logger = logger
        self.graph_relation_type = GraphRelationType.SingleRelation
        if self.tips:
            self.logger.info(self.tips)

    @property
    def tips(self):
        tips = ""
        if self.edge_class is None:
            tips += "edge_class is %s, representing simple network, " \
                    "edge_weight or edge_type specified will be invalid, " \
                    "using %s to include more information" \
                    % (None if self.edge_class is None else self.edge_class.__name__,
                       tuple(n.__name__ for n in (int, float, tuple, list) + GraphEdgeType))
        elif self.edge_class in (int, float):
            tips += "edge_class is %s, representing simple weighted network, " \
                    "edge_type specified will be invalid, " \
                    "using %s to include edge type information" \
                    % (None if self.edge_class is None else self.edge_class.__name__,
                       tuple(n.__name__ for n in (tuple, list) + GraphEdgeType))
        return tips

    @property
    def info(self):
        pass

    def add_vertex(self, vertex):
        self.vertexes.add(vertex)
        return vertex

    def add_edge(self, nodes, edge):
        if edge:
            if hasattr(edge, 'type'):
                edge_type = edge.type
            elif isinstance(edge, (tuple, list)):
                edge_type = edge[1]
            else:
                edge_type = None
            assert type(nodes) is tuple and len(nodes) == 2
            if not edge_type:
                self.edges[nodes] = edge
            else:
                self.graph_relation_type = GraphRelationType.MultiRelation
                if edge_type not in self.edges:
                    self.edges[edge_type] = {}
                self.edges[edge_type][nodes] = edge
        else:
            self.edges[nodes] = None

    def new_vertex(self, *args, **kwargs):
        if self.vertex_class:
            vertex = self.vertex_class(*args, **kwargs)
            self.add_vertex(vertex)
            return vertex
        return None

    def new_edge(self, *args, **kwargs):
        if self.edge_class:
            if self.edge_class in (int, float):
                return self.edge_class(kwargs['edge_weight'])
            elif self.edge_class in (tuple, list):
                return self.edge_class([kwargs['edge_weight'], kwargs['edge_type']])
            else:
                return self.edge_class(*args, **kwargs)
        if 'edge_weight' in kwargs and kwargs['edge_weight']:
            self.logger.debug("edge_class is %s, representing simple network, "
                              "edge_weight specified will be invalid, "
                              "using %s to include more information"
                              % (None if self.edge_class is None else self.edge_class.__name__,
                                 tuple(n.__name__ for n in (int, float, tuple, list) + GraphEdgeType)))
        return None

    def get_edge(self, node_1, node_2, edge_type=None):
        raise NotImplementedError

    def get_edges(self, node_1, node_2, edge_types=None):
        res_list = []
        if edge_types is None:
            if self.graph_relation_type is GraphRelationType.MultiRelation:
                e_types = list(self.edges.keys())
            else:
                e_types = [None]
        else:
            e_types = edge_types
        try:
            for edge_type in e_types:
                edge = self.get_edge(node_1, node_2, edge_type)
                if edge:
                    res_list.append((node_1, node_2, edge_type, edge))
            return res_list
        except Exception as e:
            logger.error(e)

    def real_edge_type(self, edge_type):
        if edge_type is not None and self.edge_class in (None, int, float):
            self.logger.debug(
                "edge_class is %s, representing simple network, "
                "edge_type specified to %s will be invalid, using %s to include edge_type information"
                % (None if self.edge_class is None else self.edge_class.__name__,
                   edge_type, tuple(n.__name__ for n in (tuple, list) + GraphEdgeType)))
            return None
        return edge_type

    def _link(self, node_1, node_2, edge):
        raise NotImplementedError

    def link(self, node_1, node_2, edge_weight=None, edge_type=None):
        edge_type = self.real_edge_type(edge_type)
        edge = self.get_edge(node_1, node_2, edge_type)
        if edge:
            self.logger.warn("link of %s and %s existed, value will not be changed" % (get_id(node_1), get_id(node_2)))
            return edge
        edge = self.new_edge((node_1, node_2), edge_weight=edge_weight, edge_type=edge_type)
        self._link(node_1, node_2, edge)
        return node_1, node_2, edge

    def del_link(self, node_1, node_2, edge_type=None):
        edges = self.get_edges(node_1, node_2, edge_type)
        if self.graph_relation_type is GraphRelationType.MultiRelation:
            for node_1, node_2, edge_type, _ in edges:
                del self.edges[edge_type][(node_1, node_2)]
        return edges

    def id_graph(self):
        v2id = {}
        vertexes = []
        for idx, vertex in enumerate(self.vertexes):
            if vertex.id:
                v2id[vertex] = vertex.id
            else:
                v2id[vertex] = idx
            vertexes.append(v2id[vertex])
        edges = {}
        if self.graph_relation_type is GraphRelationType.SingleRelation:
            for nodes, edge in self.edges.items():
                edges[(v2id[nodes[0]], v2id[nodes[1]])] = edge
        elif self.graph_relation_type is GraphRelationType.MultiRelation:
            for relation_type, nodes_edge in self.edges.items():
                if relation_type not in edges:
                    edges[relation_type] = {}
                for nodes, edge in nodes_edge.items():
                    edges[relation_type][(v2id[nodes[0]], v2id[nodes[1]])] = edge

        return vertexes, edges, self.__class__, self.graph_relation_type


class UndirectedGraph(Graph):
    def get_edge(self, node_1, node_2, edge_type=None):
        edges = self.edges.get(edge_type, None) if edge_type else self.edges
        if edges and (node_1, node_2) in self.edges:
            return node_1, node_2, edges[(node_1, node_2)]
        elif edges and (node_2, node_1) in self.edges:
            return node_2, node_1, edges[(node_2, node_1)]
        else:
            return None

    def _link(self, node_1, node_2, edge):
        node_1.linkto(node_2)
        node_2.linkto(node_1)
        self.add_edge((node_1, node_2), edge)


class DirectedGraph(Graph):
    def get_edge(self, precursor, successor, edge_type=None):
        edges = self.edges.get(edge_type, None) if edge_type else self.edges
        if edges and (precursor, successor) in self.edges:
            return precursor, successor, edges[(precursor, successor)]
        else:
            return None

    def _link(self, precursor, successor, edge):
        precursor.linkto(successor)
        self.add_edge((precursor, successor), edge)


'''
refer to https://github.com/uolcano/blog/issues/13
'''


def gen_viz_graph(graph, save_format='pdf', **kwargs):
    import graphviz
    if isinstance(graph, Graph) or (hasattr(graph, 'vertexes') and hasattr(graph, 'edges')):
        vertexes, edges, graph_type, graph_relation_type = graph.vertexes, graph.edges, \
                                                           graph.__class__, graph.graph_relation_type
    elif isinstance(graph, (tuple, list)):
        if len(graph) == 4:
            vertexes, edges, graph_type, graph_relation_type = graph
        else:
            raise GraphPlotTypeError(
                "tuple or list should have exactly 4 elements, "
                "which indicating vertexes, edges, graph_type and graph_relation_type")
    elif isinstance(graph, dict) and 'vertexes' in graph and 'edges' in graph and 'graph_type' in graph:
        vertexes, edges, graph_type, graph_relation_type = graph['vertexes'], graph['edges'], graph.get('graph_type'), \
                                                           graph.get('graph_relation_type')
    else:
        raise GraphPlotTypeError(
            "the graph type should be subclass of Graph or %s, now is %s" % ((tuple, list, dict), type(graph)))

    gv_graph = graphviz.Digraph if isinstance(graph, DirectedGraph) else graphviz.Graph
    dot = gv_graph(kwargs.get('name', 'graph'), kwargs.get('comment', ''), format=save_format)

    for idx, vertex in enumerate(vertexes):
        if isinstance(vertex, int):
            dot.node(str(vertex), str(vertex))
        elif vertex.id:
            dot.node(str(vertex), str(vertex.id))
        else:
            dot.node(str(vertex), str(idx))

    if graph_relation_type is GraphRelationType.SingleRelation:
        for nodes, edge in edges.items():
            dot.edge(str(nodes[0]), str(nodes[1]),
                     string(edge) if not hasattr(edge, 'info') or not edge.info else edge.info)
    elif graph_relation_type is GraphRelationType.MultiRelation:
        for relation_type, nodes_edge in edges.items():
            for nodes, edge in nodes_edge.items():
                dot.edge(str(nodes[0]), str(nodes[1]),
                         string(relation_type) if not hasattr(edge, 'info') or not edge.info else edge.info)
    else:
        raise GraphRelationTypeError()

    return dot


def plot_graph(graph, save_path="plot/network", save_format='pdf', view=False, **kwargs):
    dot = gen_viz_graph(
        graph=graph,
        save_format=save_format,
        **kwargs
    )
    dot.render(save_path, view=view)


if __name__ == '__main__':
    from longling.lib.structure.graph.graph_node import DirectedGraphNode

    vs = range(5)
    es = [
        (0, 1, 1, 'parent'),
        (0, 1, 1, 'parent'),
        (1, 2, 1, 'parent'),
        (2, 4, 1, 'parent'),
        (3, 4, 1, 'parent'),
        (0, 4, 2, 'friend'),
        (2, 4, 2, 'friend'),
    ]

    self = DirectedGraph(vertex_class=DirectedGraphNode, edge_class=int)
    tv = {v: self.new_vertex(None, id=v) for v in vs}
    for e in es:
        self.link(tv[e[0]], tv[e[1]], e[2], e[3])

    print(self.id_graph())
    print(gen_viz_graph(self.id_graph()).source)
    # plot_graph(graph.id_graph(), view=True)
