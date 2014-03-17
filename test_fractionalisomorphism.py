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
"""Unit tests for the :mod:`fractionalisomorphism` module.

These tests should be run with ``nose``, by executing the following command::

    nosetests

"""
from fractionalisomorphism import are_fractionally_isomorphic
from fractionalisomorphism import coarsest_equitable_partition
from fractionalisomorphism import fractionally_isomorphic_graphs
from fractionalisomorphism import graph_from_file
from fractionalisomorphism import _matrices_with_row_sums
from fractionalisomorphism import neighbors
from fractionalisomorphism import _sequences_of_ones


def test_neighbors():
    G = graph_from_file('test_graph1.txt')
    assert neighbors(G, 0) == {1, 2, 3, 6}
    assert neighbors(G, 0, set(range(6))) == {1, 2, 3}
    assert neighbors(G, 0, set(range(6, 12))) == {6}


def test_coarsest_equitable_partition():
    G = graph_from_file('test_graph1.txt')
    partition = coarsest_equitable_partition(G)
    block1 = frozenset(range(6))
    block2 = frozenset(range(6, 12))
    assert frozenset((block1, block2)) == partition


def test_are_fractionally_isomorphic():
    G = graph_from_file('test_graph1.txt')
    H = graph_from_file('test_graph2.txt')
    assert are_fractionally_isomorphic(G, G)
    assert are_fractionally_isomorphic(G, H)
    assert are_fractionally_isomorphic(H, H)


def test_sequences_of_ones():
    expected = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    actual = _sequences_of_ones(3, 2)
    assert len(expected) == len(actual)
    for x in expected:
        assert x in actual
    expected = [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
                [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]]
    actual = _sequences_of_ones(4, 2)
    assert len(expected) == len(actual)
    for x in expected:
        assert x in actual, x


def test_matrices_with_row_sums():
    matrices = list(_matrices_with_row_sums(2, 3, 2))
    assert len(matrices) == 9


def test_fractionally_isomorphic_graphs():
    G = graph_from_file('test_graph1.txt')
    H = graph_from_file('test_graph2.txt')
    isomorphic_graphs = list(fractionally_isomorphic_graphs(G))
    for graph in isomorphic_graphs:
        print(graph)
    assert G in isomorphic_graphs
    assert H in isomorphic_graphs
