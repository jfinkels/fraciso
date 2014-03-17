# graphs.py - data structures and functions for simple undirected graphs
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
from collections import namedtuple
from itertools import combinations

from fraciso.matrices import Matrix

#: A graph consisting of a set of vertices and a set of edges.
#:
#: The vertex and edge sets must be instances of :class:`frozenset`.
#: Each edge must be a :class:`frozenset` containing exactly two elements from
#: the set of vertices. For example, to create the circle graph on three
#: vertices::
#:
#:     fs = frozenset
#:     vertices = fs({1, 2, 3})
#:     edges = fs({fs({1, 2}), fs({2, 3}), fs({3, 1})})
#:     G = Graph(vertices, edges)
#:
Graph = namedtuple('Graph', ['V', 'E'])


def graph_from_file(filename):
    """Reads an adjacency matrix from a file and returns an instance of
    :data:`Graph`.

    The adjacency matrix must have entries separated by a space and rows
    separated by a newline.

    """
    with open(filename, 'r') as f:
        adjacency = [[int(b) for b in line.strip().split()]
                     for line in f.readlines()]
    return matrix_to_graph(adjacency)


def neighbors(graph, v, block=None):
    """Returns the neighbors of vertex v in the specified graph.

    If `block` is a set of vertices in the given graph, this function returns
    only the neighbors of v in the specified block. Otherwise, it returns all
    neighbors of v in the specified graph.

    """
    if block is None:
        block = graph.V
    return {w for w in block if frozenset((v, w)) in graph.E}


def graph_to_matrix(graph):
    """Returns the adjacency matrix of the specified graph, as an instance of
    :class:`Matrix`.

    """
    n = len(graph.V)
    return Matrix([[1 if frozenset((i, j)) in graph.E else 0 for j in range(n)]
                   for i in range(n)])


def matrix_to_graph(matrix):
    """Converts the given adjacency matrix to an instance of :data:`Graph`."""
    n = len(matrix)
    vertices = frozenset(range(n))
    # We can use combinations since we're considering only undirected graphs.
    edges = frozenset({frozenset((u, v)) for u, v in combinations(vertices, 2)
                       if matrix[u][v] == 1})
    return Graph(vertices, edges)
