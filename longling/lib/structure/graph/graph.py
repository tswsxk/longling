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

    @staticmethod
    def is_multi_relation(relation_type):
        if relation_type is GraphRelationType.SingleRelation:
            return False
        elif relation_type is GraphRelationType.MultiRelation:
            return True


class GraphRelationTypeError(TypeError):
    pass


class GraphPlotTypeError(TypeError):
    pass


EDGE_CLASS = (None, int, float, tuple, list) + GraphEdgeType
VERTEX_CLASS = (None,) + GraphNodeType


class Graph(object):
    def __init__(self, vertexes_class=AddList, edges_class=dict, vertex_class=None, edge_class=None, logger=logger):
        """
        图结构
        Parameters
        ----------
        vertexes_class:
            A structure to hold all vertexes.
        edges_class:
            A structure to hold all edges.
        vertex_class: VERTEX_CLASS
            A structure to generate new vertex.
        edge_class: EDGE_CLASS
            A structure to generate new edge.
            When edge class is None, it is a simple graph without weight and relation type;
            When edge class in (int float), the graph is a simple weighted graph without relation type;
            In other conditions, when relation_type is specified, the graph will become multi-relational graph
            If there are several relation, which can be refered by querying attribute 'graph_relation_type'.
            In visualization, the relation will be first prior to be shown while the weight next.
            More information to be shown in graph can be specified in the 'info' attribute of the edge_class.
        logger: logging.logger
        """
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
        """
        初始化的提示信息
        Returns
        -------

        """
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

    def __str__(self):
        return "%s\n%s" % (self.vertexes, self.edges)

    def add_vertex(self, vertex):
        """
        添加新的节点
        Parameters
        ----------
        vertex: VERTEX_CLASS
        Returns
        -------

        """
        self.vertexes.add(vertex)
        return vertex

    def add_edge(self, nodes, edge):
        """
        添加新的边
        Parameters
        ----------
        nodes: tuple(VERTEX_CLASS, VERTEX_CLASS)
        edge: EDGE_CLASS

        Returns
        -------

        """
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
        """
        生成新的顶点
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        if self.vertex_class:
            vertex = self.vertex_class(*args, **kwargs)
            self.add_vertex(vertex)
            return vertex
        return None

    def new_edge(self, *args, **kwargs):
        """
        生成新的边
        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
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
        """
        获取node_1,node_2之间的某种连边
        Parameters
        ----------
        node_1: VERTEX_CLASS
        node_2: VERTEX_CLASS
        edge_type:

        Returns
        -------

        """
        raise NotImplementedError

    def get_edges(self, node_1, node_2, edge_types=None):
        """
        获取node_1,node_2之间的所有符合条件的连边
        Parameters
        ----------
        node_1
        node_2
        edge_types: None or iterable

        Returns
        -------

        """
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
        """
        统一边类型
        Parameters
        ----------
        edge_type

        Returns
        -------

        """
        if edge_type is not None and self.edge_class in (None, int, float):
            self.logger.debug(
                "edge_class is %s, representing simple network, "
                "edge_type specified to %s will be invalid, using %s to include edge_type information"
                % (None if self.edge_class is None else self.edge_class.__name__,
                   edge_type, tuple(n.__name__ for n in (tuple, list) + GraphEdgeType)))
            return None
        return edge_type

    def _link(self, node_1, node_2, edge):
        """
        将 node_1, node_2 通过 edge 连起来
        Parameters
        ----------
        node_1: VERTEX_CLASS
        node_2: VERTEX_CLASS
        edge: EDGE_CLASS

        Returns
        -------

        """
        raise NotImplementedError

    def link(self, node_1, node_2, edge_weight=None, edge_type=None):
        """
        将 node_1, node_2 连起来
        边的权重为 edge_weight，边的类型为 edge_type
        Parameters
        ----------
        node_1: VERTEX_CLASS
        node_2: VERTEX_CLASS
        edge_weight: int or float or None
        edge_type:

        Returns
        -------

        """
        edge_type = self.real_edge_type(edge_type)
        edge = self.get_edge(node_1, node_2, edge_type)
        if edge:
            self.logger.warn("link of %s and %s existed, value will not be changed" % (get_id(node_1), get_id(node_2)))
            return edge
        edge = self.new_edge((node_1, node_2), edge_weight=edge_weight, edge_type=edge_type)
        self._link(node_1, node_2, edge)
        return node_1, node_2, edge

    def del_link(self, node_1, node_2, edge_type=None):
        """
        删除某两个节点 node_1, node_2 的某关系连边
        Parameters
        ----------
        node_1: VERTEX_CLASS
        node_2: VERTEX_CLASS
        edge_type

        Returns
        -------

        """
        edges = self.get_edges(node_1, node_2, edge_type)
        if self.graph_relation_type is GraphRelationType.MultiRelation:
            for node_1, node_2, edge_type, _ in edges:
                del self.edges[edge_type][(node_1, node_2)]
        return edges

    def is_multi_edge_type(self):
        return GraphRelationType.is_multi_relation(self.graph_relation_type)

    def id_graph(self, edge2id=None, force_v2idx=False):
        """
        替换vertexes里的所有节点为节点id，生成新的id标识的图，不改变原有图结构
        Parameters
        ----------
        edge2id: None or bool or dict
            When specified, transform the edge to edge id
        force_v2idx: bool
            When True, force the id become index
        Returns
        -------

        """
        v2id = {}
        vertexes = []
        for idx, vertex in enumerate(self.vertexes):
            if vertex.id and not force_v2idx:
                v2id[vertex] = vertex.id
            else:
                v2id[vertex] = idx
            vertexes.append(v2id[vertex])
        edges = {}
        if edge2id is True:
            edge2id = self.edge_type_id

        if edge2id:
            def rel_type_map(rel_type):
                return edge2id[rel_type]
        else:
            def rel_type_map(rel_type):
                return rel_type

        if self.graph_relation_type is GraphRelationType.SingleRelation:
            for nodes, edge in self.edges.items():
                edges[(v2id[nodes[0]], v2id[nodes[1]])] = edge
        elif self.graph_relation_type is GraphRelationType.MultiRelation:
            for relation_type, nodes_edge in self.edges.items():
                relation_type = rel_type_map(relation_type)
                if relation_type not in edges:
                    edges[relation_type] = {}
                for nodes, edge in nodes_edge.items():
                    edges[relation_type][(v2id[nodes[0]], v2id[nodes[1]])] = edge

        return vertexes, edges, self.__class__, self.graph_relation_type

    @property
    def vertex_num(self):
        return len(self.vertexes)

    @property
    def edge_type_id(self):
        if self.is_multi_edge_type():
            return {edge_type: idx for idx, edge_type in enumerate(self.edges.keys())}
        else:
            return None

    def _get_adjacent_matrix(self, keep_edge_dim=False, directed=False):
        """
        获取图对应的邻接矩阵
        Parameters
        ----------
        keep_edge_dim: bool
            是否在关系只有一种时,保留关系维度
            为真时:  N * N * 1
            为假时: N * N
        directed: bool
            是否为有向图
        Returns
        -------
        """
        vertexes, edges, _, _ = self.id_graph(edge2id=True, force_v2idx=True)

        import numpy as np

        n = len(vertexes)

        if self.is_multi_edge_type():
            edge_dim = len(edges.keys())
            if edge_dim == 1 and not keep_edge_dim:
                adjacent_matrix = np.zeros((n, n))
                edges = edges.values()
            else:
                adjacent_matrix = np.zeros((n, n, edge_dim))
        else:
            if not keep_edge_dim:
                adjacent_matrix = np.zeros((n, n))
            else:
                adjacent_matrix = np.zeros((n, n, 1))

        if len(np.shape(adjacent_matrix)) == 3:
            for edge_id, edge in edges.items():
                for (head, tail), weight in edge.items():
                    weight = 1 if weight is None else weight
                    adjacent_matrix[head][tail][edge_id] = weight
                    if not directed:
                        adjacent_matrix[tail][head][edge_id] = weight

        elif len(np.shape(adjacent_matrix)) == 2:
            for (head, tail), weight in edges.items():
                weight = 1 if weight is None else weight
                adjacent_matrix[head][tail] = weight
                if not directed:
                    adjacent_matrix[tail][head] = weight

        else:
            raise AssertionError("adjacent_matrix must be 3 dim or 2 dim")

        return adjacent_matrix

    def get_adjacent_matrix(self, keep_edge_dim=False):
        raise NotImplementedError

    def load_from_adjacent_matrix(self, adjacent_matrix, vertexes=None, edge_types=None):
        """
        从邻接矩阵中构建图
        Parameters
        ----------
        adjacent_matrix: list or np.array or np.matrix
            邻接矩阵
        vertexes: None or VERTEX_CLASS
            顶点集合
        edge_types: list or dict
            边类型集合
        Returns
        -------

        """
        if vertexes is None:
            for i in range(len(adjacent_matrix)):
                self.new_vertex(i)
        else:
            for vertex in vertexes:
                self.add_vertex(vertex)

        if edge_types:
            def get_edge_type(idx):
                return edge_types[idx]
        else:
            def get_edge_type(idx):
                return idx
        for head, tail_weight in enumerate(adjacent_matrix):
            for tail, weight in enumerate(tail_weight):
                if hasattr(weight, "len"):
                    for w_idx, weight in enumerate(weight):
                        if weight:
                            self.link(self.vertexes[head], self.vertexes[tail], weight, get_edge_type(w_idx))
                else:
                    if weight:
                        self.link(self.vertexes[head], self.vertexes[tail], weight)


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

    def get_adjacent_matrix(self, keep_edge_dim=False):
        return self._get_adjacent_matrix(keep_edge_dim, False)


class DirectedGraphLoopError(Exception):
    pass


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

    def get_adjacent_matrix(self, keep_edge_dim=False):
        return self._get_adjacent_matrix(keep_edge_dim, True)

    def get_heads(self):
        """
        获取所有入度为0的点
        Returns
        -------

        """
        heads = []
        for vertex in self.vertexes:
            if vertex.in_degree == 0:
                heads.append(vertex)
        return heads

    def get_tails(self):
        """
        获取所有出度为0的点
        Returns
        -------

        """
        tails = []
        for vertex in self.vertexes:
            if vertex.out_degree == 0:
                tails.append(vertex)
        return tails

    def toposort(self, level=False, merge_tag=True, reversed=False):
        """
        暂时只支持单/全关系的拓扑排序，不支持特定关系下的关系排序
        Parameters
        ----------
        level: bool
            层次排序
        merge_tag: bool
            是否将输出合并为一个列表，只在 level 为 True 时有效
        reversed: bool
            是否进行反向层次排序，只在 level 为 True 时有效
        Returns
        -------

        """
        from copy import deepcopy
        vertexes = deepcopy(self.vertexes)
        id2vertexes = {repr(true_vertex): true_vertex for true_vertex in self.vertexes}
        true_vertexes = {repr(vertex): id2vertexes[repr(vertex)] for vertex in vertexes}

        toposort_list = []
        visited = set()
        for vertex in vertexes:
            vertex.successor = set(vertex.successor)
            vertex.precursor = set(vertex.precursor)
        if not level:
            loop_tag = True
            while loop_tag:
                loop_tag = False
                for vertex in vertexes:
                    if vertex not in visited and vertex.in_degree == 0:
                        visited.add(vertex)
                        toposort_list.append(true_vertexes[repr(vertex)])
                        for successor in vertex.successor:
                            successor.precursor.remove(vertex)
                    elif vertex not in visited:
                        loop_tag = True
        elif not reversed:
            loop_tag = True
            while loop_tag:
                level_sort = []
                move_dict = {}
                loop_tag = False
                for vertex in vertexes:
                    if vertex not in visited and vertex.in_degree == 0:
                        visited.add(vertex)
                        level_sort.append(true_vertexes[repr(vertex)])
                        move_dict[vertex] = vertex.successor
                    elif vertex not in visited:
                        loop_tag = True
                toposort_list.append(level_sort)
                for vertex, successor_list in move_dict.items():
                    for successor in successor_list:
                        successor.precursor.remove(vertex)
        else:
            loop_tag = True
            while loop_tag:
                level_sort = []
                move_dict = {}
                loop_tag = False
                for vertex in vertexes:
                    if vertex not in visited and vertex.out_degree == 0:
                        visited.add(vertex)
                        level_sort.append(true_vertexes[repr(vertex)])
                        move_dict[vertex] = vertex.precursor
                    elif vertex not in visited:
                        loop_tag = True
                toposort_list.append(level_sort)
                for vertex, precursor_list in move_dict.items():
                    for precursor in precursor_list:
                        precursor.successor.remove(vertex)
            toposort_list.reverse()
        if level and merge_tag:
            tmp = []
            for level_sort in toposort_list:
                tmp.extend(level_sort)
            toposort_list = tmp
        if not level or merge_tag:
            if len(toposort_list) != self.vertex_num:
                raise DirectedGraphLoopError
        else:
            if sum([len(level_sort) for level_sort in toposort_list]) != self.vertex_num:
                raise DirectedGraphLoopError

        return toposort_list

    def isloop(self):
        """
        检查图中是否有回路
        Returns
        -------

        """
        try:
            self.toposort()
        except DirectedGraphLoopError:
            return True
        return False


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


from mxnet import gluon


class GraphNN(gluon.Block):
    def __init__(self, graph: DirectedGraph, graph_node_num, dim=256, **kwargs):
        super(GraphNN, self).__init__(**kwargs)
        self.graph = graph
        self.graph_node_num = graph_node_num
        with self.name_scope():
            self.graph_embedding = gluon.nn.Embedding(graph_node_num, dim)
            for i in range(self.graph.vertex_num):
                setattr(self, "node%s", gluon.nn.Dense(dim))
        self.visit_seq = self.graph.toposort(level=True, merge_tag=False, reversed=True)

    def forward(self, whole_node, *args):
        node_embeddings = self.graph_embedding(whole_node)
        for visits in self.visit_seq:


    def graph_viz(self):
        plot_graph(self.graph, view=True)


if __name__ == '__main__':
    from longling.lib.structure.graph.graph_node import DirectedGraphNode, UndirectedGraphNode

    vs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34]

    es = [[2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [4, 3], [5, 1], [6, 1], [7, 1], [7, 5], [7, 6], [8, 1], [8, 2],
          [8, 3], [8, 4], [9, 1], [9, 3], [10, 3], [11, 1], [11, 5], [11, 6], [12, 1], [13, 1], [13, 4], [14, 1],
          [14, 2], [14, 3], [14, 4], [17, 6], [17, 7], [18, 1], [18, 2], [20, 1], [20, 2], [22, 1], [22, 2], [26, 24],
          [26, 25], [28, 3], [28, 24], [28, 25], [29, 3], [30, 24], [30, 27], [31, 2], [31, 9], [32, 1], [32, 25],
          [32, 26], [32, 29], [33, 3], [33, 9], [33, 15], [33, 16], [33, 19], [33, 21], [33, 23], [33, 24], [33, 30],
          [33, 31], [33, 32], [34, 9], [34, 10], [34, 14], [34, 15], [34, 16], [34, 19], [34, 20], [34, 21], [34, 23],
          [34, 24], [34, 27], [34, 28], [34, 29], [34, 30], [34, 31], [34, 32], [34, 33]]

    graph = DirectedGraph(vertex_class=DirectedGraphNode, edge_class=None)
    tv = {v: graph.new_vertex(None, id=v) for v in vs}
    for e in es:
        graph.link(tv[e[0]], tv[e[1]])

    print(graph.id_graph())
    # print(gen_viz_graph(graph.id_graph()).source)
    plot_graph(graph, view=True)
    adjacent_matrix = graph.get_adjacent_matrix()
    print(adjacent_matrix.tolist())
    # new_graph = UndirectedGraph(vertex_class=UndirectedGraphNode, edge_class=int)
    # new_graph.load_from_adjacent_matrix(adjacent_matrix)
    # print(new_graph.get_adjacent_matrix())
    # assert new_graph.get_adjacent_matrix().all() == adjacent_matrix.all()
    print(repr(graph.get_heads()))
    print(repr(graph.get_tails()))
    print(repr(graph.toposort(level=True, merge_tag=False, reversed=True)))
