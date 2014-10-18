# partitions.py - functions for modifying partitions of a graph
#
# Copyright 2014 Jeffrey Finkelstein.
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
"""Functions for computing and modifying partitions of a graph.

A **partition** of a graph is a collection of sets of vertices that are
pairwise disjoint and whose union equals the set of all vertices in the graph.

"""
from itertools import accumulate
from itertools import chain
from itertools import combinations
from itertools import product
from itertools import groupby
from functools import reduce

import numpy as np

from fraciso.matrices import permutation_matrices


def peek(s):
    """Returns but does not remove an arbitrary element from the specified set.

    """
    return next(iter(s))


def union(*sets):
    """Returns the union of all sets given as positional arguments.

    If there is exactly one positional argument, that set itself is returned.

    If there are no positional arguments, the empty set is returned.

    """
    # The last argument to reduce is the initializer used in the case of an
    # empty sequence of sets. I would like to use ``{}`` there, but Python
    # interprets that expression as a dictionary literal instead of a set
    # literal.
    #
    # This is essentially the same as ``sets[0].union(*sets[1:])``, but it
    # works even if `sets` is not a list and if `sets` has only one element.
    return reduce(lambda S, T: S | T, sets, set())


def are_pairwise_disjoint(*sets):
    """Returns ``True`` if and only if each pair of sets is disjoint."""
    return all(S.isdisjoint(T) for S, T in combinations(sets, 2))


def is_valid_partition(graph, partition):
    """Returns ``True`` if and only if the specified partition is a valid
    partition of the vertices of `graph`.

    `graph` is an instance of :class:`networkx.Graph`.

    `partition` is a set of sets of vertices from the graph.

    A partition is valid if the set of vertices of the graph equals the
    disjoint union of the blocks in the partition.

    """
    return (are_pairwise_disjoint(*partition)
            and union(*partition) == set(graph.nodes())
            and all(len(block) > 0 for block in partition))


def degree(graph, v, block=None):
    """Returns the number of neighbors of vertex `v` in the specified graph.

    `graph` is an instance of :class:`networkx.Graph`.

    If `block` is a set of vertices in the given graph, this function returns
    only the number of neighbors in the specified block. Otherwise, it simply
    returns the degree of `v` in `graph`.

    """
    if block is None:
        block = set(graph)
    subgraph = graph.subgraph(block | {v})
    return subgraph.degree(v)


def is_block_equitable(graph, partition, block):
    """Returns ``True`` if and only if the specified block of the given
    partition is equitable.

    `graph` must be an instance of :class:`networkx.Graph`.

    `partition` is a set of sets. Each inner set, a **block**, is a set of
    vertices in the graph.

    `block` must be an element of `partition`. This block will be checked for
    equitability.

    A block is **equitable** if for each two vertices v and w in the block, the
    number of neighbors of v in block B equals the number of neighbors of w in
    block B, for all blocks B.

    """
    return all(degree(graph, v, B) == degree(graph, w, B)
               for v, w in product(block, block) for B in partition)


def is_partition_equitable(graph, partition):
    """Returns ``True`` if and only if the specified partition is an equitable
    partition of the graph.

    `graph` must be an instance of :class:`networkx.Graph`.

    `partition` is a set of sets. Each inner set, a **block**, is a set of
    vertices in the graph.

    A partition is **equitable** if for each block P1 and for each two vertices
    v and w in P1, the number of neighbors of v in block P2 equals the number
    of neighbors of w in block P2, for all blocks P2.

    """
    return all(is_block_equitable(graph, partition, block)
               for block in partition)


def coarsest_equitable_partition(graph):
    """Returns the coarsest equitable partition of the specified graph.

    `graph` must be an instance of :class:`networkx.Graph`.

    A partition is a set of sets. Each of the inner sets is a
    :class:`frozenset` of vertices in the graph.

    A partition is **equitable** if for each block P1 and for each two vertices
    v and w in P1, the number of neighbors of v in block P2 equals the number
    of neighbors of w in block P2, for all blocks P2.

    A partition P1 is coarser than partition P2 if each block of P2 is a subset
    of a block of P1. A partition is a coarsest equitable partition if it is
    equitable and no coarser partition is equitable.

    **Implementation note**: this function applies the "canonical color
    refinement" algorithm described in the 2013 paper **`Tight Lower and Upper
    Bounds for the Complexity of Canonical Colour Refinement`_** by Berkholz,
    Bonsma, and Grohe. If implemented correctly, this algorithm has a
    worst-case running time of **O((m + n) log n)**, where **m** is the number
    of edges in the graph and **n** the number of vertices. (This algorithm
    works for both directed and undirected graphs.)

    .. _color-refinement: http://dx.doi.org/10.1007/978-3-642-40450-4_13

    __ color-refinement_

    """
    num_vertices = len(G)
    # In the canonical color refinement algorithm, the blocks of the partition
    # are interpreted as colors. Initially, all vertices have the same color
    # (that is, the partition is the unit partition). Then, vertices of one
    # color are iteratively recolored according to the number of neighbors of
    # each vertex.
    #
    # Create a stack that will maintain colors that should be still be used as
    # a refining color. Initially, there is just one color, the color 1 (and
    # all vertices have this color).
    #
    # TODO rename S_refine to colors_to_be_refined
    S_refine = [(0, list(G))]
    # The algorithm terminates when there are no more colors to be refined.
    while len(S_refine) > 0:
        # Pop the next color `r` and the vertices with that color from the
        # stack.
        r, r_vertices = S_refine.pop()
        # Maintain an array of the `r`-degree of each vertex (the number of
        # neighbors of color `r` that a vertex has). Initialize it to 0.
        cdeg = [0 for n in num_vertices]
        # Compute the `r`-degrees of each vertex.
        for v in r_vertices:
            for w in G:
                if G.has_edge(w, v):
                    cdeg[w] += 1
    # TODO complete this!!!


def partition_parameters(graph, partition, as_matrices=False):
    """Returns the parameters of the given partition.

    `graph` must be an instance of :class:`networkx.Graph`.

    `partition` must be a valid equitable partition of the specified graph.

    The parameters of the graph are two dictionaries. The first dictionary is a
    mapping from a block of the partition to the number of vertices in that
    block. The second dictionary is a two-dimensional dictionary. Both keys are
    blocks of the partition. The entry indexed by blocks **i** and **j** is the
    number of neighbors in block **j** of any fixed vertex in block **i** (the
    particular vertex doesn't matter, since the partition is equitable).

    If `as_matrices` is ``True``, then the parameters will be converted to
    NumPy matrices.

    """
    vertices_per_block = {block: len(block) for block in partition}
    block_neighbors = {b_i: {b_j: degree(graph, peek(b_i), b_j)
                             for b_j in partition}
                       for b_i in partition}
    if not as_matrices:
        return vertices_per_block, block_neighbors
    return _as_vector(vertices_per_block), _as_matrix(block_neighbors)


def partition_to_permutation(graph, partition):
    """Converts the specified partition of the graph to a permutation matrix,
    representing the permutation of the adjacency matrix of the graph that
    places rows into contiguous blocks representing blocks of the partition.

    `graph` must be an instance of :class:`networkx.Graph`. Furthermore, the
    **n** vertices of the graph must be the first **n** nonnegative integers.

    `partition` must be a valid partition of `graph`.

    """
    n = len(graph)
    # Sort the blocks of the partition according to lexicographic order, and
    # sort the set of blocks as well so that the output is well-defined.
    sorted_partition = sorted(sorted(block) for block in partition)
    extents = list(accumulate(len(block) for block in sorted_partition))
    # Flatten the partition so that we have simply a sequence of vertices
    # representing a permutation.
    permutation = chain(*sorted_partition)
    # Create the permutation matrix from the permutation.
    matrix = np.matrix([[1 if j == v else 0 for j in range(n)]
                        for v in permutation])
    return matrix, extents


def lexicographic_blocks(dictionary):
    """Sorts the key/value pairs of `dictionary` according to the lexicographic
    order of the keys.

    This is useful if your keys are instances of :class:`set` and you wish them
    to be ordered according to their list representation instead of ordered
    according to inclusion.

    """
    return sorted(dictionary.items(), key=lambda item: sorted(item[0]))


def _as_vector(sizes):
    """Converts the given dictionary to a vector.

    The dictionary `sizes` maps blocks of a partition to number of vertices in
    that partition. Each element of the returned list is the number of vertices
    in a block of the partition. The list is sorted according to the
    lexicographic ordering of the blocks.

    The returned

    """
    return np.mat([size for block, size in lexicographic_blocks(sizes)]).T


def _as_matrix(block_neighbors):
    """Converts the given dictionary to a matrix.

    The two-dimensional dictionary `block_neighbors` is indexed in each
    dimension by blocks of a partition. The entry at row **i**, column **j** is
    an integer representing the number of neighbors in block **j** of any fixed
    vertex in block **i**. Each row and each column of the matrix is sorted
    according to lexicographic order of the blocks.

    """
    return np.mat([[num for block_j, num in lexicographic_blocks(d)]
                   for block_i, d in lexicographic_blocks(block_neighbors)])


def are_common_partitions(graph1, partition1, graph2, partition2):
    """Returns ``True`` if and only if the two partitions have the same
    parameters (or a permutation of the same parameters).

    `graph1` and `graph2` must be instances of :class:`networkx.Graph`.

    `partition1` and `partition2` must be valid equitable partitions of the
    corresponding graphs.

    The partition parameters are described in the documentation for
    :func:`partition_parameters`. The parameters include one vector and one
    matrix. This function returns ``True`` exactly when there is some
    permutation such that both the vector and the matrix of `partition1` equal
    the vector and the matrix of `partition2`.

    """
    # By specifying `as_matrices=True`, `sizes` becomes a vector of length p
    # and `block_neighbors` a matrix of size p by p, where p is the number of
    # blocks in the partitions. The innermost entries of both should be
    # integers.
    sizes1, neighbors1 = partition_parameters(graph1, partition1,
                                              as_matrices=True)
    sizes2, neighbors2 = partition_parameters(graph2, partition2,
                                              as_matrices=True)
    # Return True if there is any permutation that makes these two pairs equal.
    # Need to permute both the rows and the columns of the `neighbors` matrix,
    # so we multiply by the permutation on the left and its inverse on the
    # right.
    match = lambda P: (np.array_equal(sizes1, P * sizes2)
                       and np.array_equal(neighbors1, P * neighbors2 * P.I))
    return any(match(P) for P in permutation_matrices(len(sizes1)))
