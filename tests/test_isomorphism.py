# test_isomorphism.py - unit tests for fraciso.isomorphism
#
# Copyright 2014 Jeffrey Finkelstein.
#
# This file is part of fractionalisomorphism.
#
# fractionalisomorphism is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# fractionalisomorphism is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# fractionalisomorphism.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`fraciso.isomorphism` module."""
from fraciso.isomorphism import are_fractionally_isomorphic
from fraciso.isomorphism import fractionally_isomorphic_graphs
from fraciso.graphs import graph_from_file

from tests.helpers import path_to


def test_are_fractionally_isomorphic():
    G = graph_from_file(path_to('data/test_graph1.txt'))
    H = graph_from_file(path_to('data/test_graph2.txt'))
    assert are_fractionally_isomorphic(G, G)
    assert are_fractionally_isomorphic(G, H)
    assert are_fractionally_isomorphic(H, H)


def test_fractionally_isomorphic_graphs():
    G = graph_from_file(path_to('data/test_graph1.txt'))
    H = graph_from_file(path_to('data/test_graph2.txt'))
    isomorphic_graphs = list(fractionally_isomorphic_graphs(G))
    for graph in isomorphic_graphs:
        print(graph)
    assert G in isomorphic_graphs
    assert H in isomorphic_graphs
