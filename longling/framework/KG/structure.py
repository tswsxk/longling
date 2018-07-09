# coding:utf-8
# created by tongshiwei on 2018/6/29

from longling.lib.structure.graph.graph import DirectedGraph,gen_viz_graph, plot_graph


class RDFGraph(DirectedGraph):
    def add_rdf(self, rdf_triple, formulation_func=lambda x: x, is_edge_e_type=True):
        rdf_subject, rdf_edge, rdf_object = formulation_func(rdf_triple)
        if rdf_subject not in self.vertexes:
            self.logger.info("add new vertex %s" % rdf_subject)
            subject_node = self.add_vertex(rdf_subject)
        else:
            subject_node = rdf_subject

        if rdf_object not in self.vertexes:
            self.logger.info("add new vertex %s" % rdf_object)
            object_node = self.add_vertex(rdf_object)
        else:
            object_node = rdf_object

        if is_edge_e_type:
            edge_node = self.get_edge(subject_node, object_node, rdf_edge)
            if not edge_node:
                self.logger.info("add new edge (%s, %s)-%s" % (subject_node, object_node, edge_node))
                self.link(subject_node, object_node, edge_type=rdf_edge)
        else:
            if rdf_edge not in self.get_edges(subject_node, object_node):
                self.link(subject_node, object_node, *rdf_edge)


if __name__ == '__main__':
    from longling.lib.structure.graph.graph_node import DirectedGraphNode

    rdf_triples = [
        (0, 'parent', 1),
        (0, 'parent', 1),
        (1, 'parent', 2),
        (2, 'parent', 4),
        (3, 'parent', 4),
        (0, 'friend', 4),
        (2, 'friend', 4),
    ]
    vs = {}
    for rdf_triple in rdf_triples:
        for i in {0, 2}:
            if rdf_triple[i] not in vs:
                vs[rdf_triple[i]] = DirectedGraphNode(None, id=rdf_triple[i])
    rdf_triples = [(vs[rdf_triple[0]], rdf_triple[1], vs[rdf_triple[2]]) for rdf_triple in rdf_triples]

    graph = RDFGraph(vertex_class=DirectedGraphNode, edge_class=tuple)

    for rdf_triple in rdf_triples:
        graph.add_rdf(
            rdf_triple,
            is_edge_e_type=True,
        )

    # print(graph.id_graph())
    # print(gen_viz_graph(graph.id_graph()).source)
    # plot_graph(graph, view=True)