# test_isomorphism.py - unit tests for fraciso.isomorphism
#
# Copyright 2014 Jeffrey Finkelstein.
#
# This file is part of fractionalisomorphism.
#
# This file is part of fraciso.
#
# fraciso is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# fraciso is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# fraciso.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`fraciso.isomorphism` module."""
from fraciso.isomorphism import are_fractionally_isomorphic
from fraciso.isomorphism import random_fractionally_isomorphic_graph
from fraciso.isomorphism import fractionally_isomorphic_graphs

from .helpers import graph_from_file
from .helpers import skip


def test_are_fractionally_isomorphic():
    G = graph_from_file('data/test_graph1.txt')
    H = graph_from_file('data/test_graph2.txt')
    assert are_fractionally_isomorphic(G, G)
    assert are_fractionally_isomorphic(G, H)
    assert are_fractionally_isomorphic(H, H)


def test_random_fractionally_isomorphic_graph():
    G = graph_from_file('data/test_graph1.txt')
    for seed in (123, 456, 789):
        H = random_fractionally_isomorphic_graph(G, seed=seed)
        assert are_fractionally_isomorphic(G, H)


@skip('fractionally_isomorphic_graphs() is not implemented')
def test_fractionally_isomorphic_graphs():
    G = graph_from_file('data/test_graph1.txt')
    H = graph_from_file('data/test_graph2.txt')
    isomorphic_graphs = list(fractionally_isomorphic_graphs(G))
    assert G in isomorphic_graphs
    assert H in isomorphic_graphs
