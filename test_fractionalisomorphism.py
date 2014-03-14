from fractionalisomorphism import graph_from_file
from fractionalisomorphism import are_fractionally_isomorphic


def test_are_fractionally_isomorphic():
    G = graph_from_file('test_graph1.txt')
    H = graph_from_file('test_graph2.txt')
    assert are_fractionally_isomorphic(G, H)
