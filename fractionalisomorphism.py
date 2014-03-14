#!/usr/bin/env python3
"""Algorithms for determining if two graphs are fractionally isomorphic.

Example usage::

    from fractionalisomorphism import Graph
    from fractionalisomorphism import are_fractionally_isomorphic

    # Create a graph G.
    vertices = {1, 2, 3}
    edges = {frozenset(1, 2), frozenset(2, 3), frozenset(3, 1)}
    G = Graph(vertices, edges)

    # Create a graph H.
    #H = ...

    # Determine if they are fractionally isomorphic.
    if are_fractionally_isomorphic(G, H):
        print('Yes')
    else:
        print('No')

"""
from collections import namedtuple
from functools import reduce
from itertools import combinations
from itertools import groupby
from itertools import permutations
from itertools import product

__all__ = ['Graph', 'are_fractionally_isomorphic']


def union(*sets):
    """Returns the union of all sets given as positional arguments."""
    # The last argument to reduce is the initializer used in the case of an
    # empty sequence of sets.
    #
    # This is essentially the same as ``sets[0].union(*sets[1:])``, but it
    # works even if `sets` is not a list and if `sets` has only one element.
    return reduce(lambda S, T: S | T, sets, {})


def are_pairwise_disjoint(*sets):
    """Returns ``True`` if and only if each pair of sets is disjoint."""
    return all(S.isdisjoint(T) for S, T in combinations(sets, 2))


def is_valid_partition(graph, partition):
    """Returns ``True`` if and only if the specified partition is a valid
    partition of the vertices of `graph`.

    A partition is valid if the set of vertices of the graph equals the
    disjoint union of the blocks in the partition.

    """
    return are_pairwise_disjoint(*partition) and union(*partition) == graph.V \
        and all(len(block) > 0 for block in partition)


def neighbors(graph, v, block=None):
    """Returns the neighbors of vertex v in the specified graph.

    If `block` is a set of vertices in the given graph, this function returns
    only the neighbors of v in the specified block. Otherwise, it returns all
    neighbors of v in the specified graph.

    """
    if block is None:
        block = graph.V
    return {w for w in block if (v, w) in graph.E}


def is_block_equitable(graph, partition, block):
    """Returns ``True`` if and only if the specified block of the given
    partition is equitable.

    `graph` must be an instance of :data:`Graph`.

    `partition` is a set of sets. Each inner set, a **block**, is a set of
    vertices in the graph.

    `block` must be an element of `partition`. This block will be checked for
    equitability.

    A block is **equitable** if for each two vertices v and w in the block, the
    number of neighbors of v in block B equals the number of neighbors of w in
    block B, for all blocks B.

    """
    return all(len(neighbors(graph, v, B)) == len(neighbors(graph, w, B))
               for v, w in product(block, block) for B in partition)


def is_partition_equitable(graph, partition):
    """Returns ``True`` if and only if the specified partition is an equitable
    partition of the graph.

    `graph` must be an instance of :data:`Graph`.

    `partition` is a set of sets. Each inner set, a **block**, is a set of
    vertices in the graph.

    A partition is **equitable** if for each block P1 and for each two vertices
    v and w in P1, the number of neighbors of v in block P2 equals the number
    of neighbors of w in block P2, for all blocks P2.

    """
    return all(is_block_equitable(graph, partition, block)
               for block in partition)


def _adapt(graph, partition):
    """Adapt the specified equitable partition into the coarsest equitable one.

    `graph` must be an instance of :data:`Graph`.

    `partition` is a set of blocks, each of which is a :class:`frozenset` of
    vertices from the graph.

    """
    # First, for each vertex, compute the number of neighbors of that vertex in
    # each block (including its own).
    block_neighbors = {v: {block: len(neighbors(graph, v, block))
                           for block in partition}
                       for v in graph.V}
    # Second, partition all vertices into new blocks consisting of vertices
    # that have the same dictionary of block neighbors as computed above.
    #
    # We do this by first sorting all the vertices according to their block
    # neighbors dictionary, then grouping together the vertices that have the
    # same block neighbors dictionary. However, since dictionaries cannot be
    # used as sort keys (because they are not comparable), we use the sorted
    # list of key/value pairs. Even more annoying, the keys in these key/value
    # pairs are themselves sets, which must be converted to sorted lists
    # (because the default comparison on sets in Python is the "is subset"
    # comparison).
    #
    # This is quite an inefficient algorithm for doing this, but the reward is
    # readability to match the mathematical description of the algorithm more
    # closely.
    key_func = lambda w: sorted((sorted(k), v)
                                for k, v in block_neighbors[w].items())
    vertices = sorted(block_neighbors, key=key_func)
    # Next, we simply group the vertices according to ones that have the same
    # block neighbors dictionary.
    new_partition = {frozenset(g) for k, g in groupby(vertices, key_func)}

    # Do a sanity check...
    #assert is_valid_partition(new_partition)

    # If this process produced exactly the same partition, then we have reached
    # the base case of the recursion.
    if new_partition == partition:
        return partition
    return _adapt(graph, new_partition)


def coarsest_equitable_partition(graph):
    """Returns the coarsest equitable partition of the specified graph.

    `graph` must be an instance of :data:`Graph`.

    A partition is a set of sets. Each of the inner sets is a
    :class:`frozenset` of vertices in the graph.

    A partition is **equitable** if for each block P1 and for each two vertices
    v and w in P1, the number of neighbors of v in block P2 equals the number
    of neighbors of w in block P2, for all blocks P2.

    A partition P1 is coarser than partition P2 if each block of P2 is a subset
    of a block of P1. A partition is a coarsest equitable partition if it is
    equitable and no coarser partition is equitable.

    """
    # Start with the partition that consists of a single block containing the
    # entire vertex set. Recursively adapt the partition until no further
    # adaptations are possible. This is guaranteed to be the coarsest equitable
    # partition.
    return _adapt(graph, {graph.V})


def partition_parameters(graph, partition):
    # This ensures that every list comprehension iterates over the blocks in
    # the same order.
    partition_as_list = list(partition)
    # pre-condition: each block is regular
    vertices_per_block = [len(block) for block in partition_as_list]
    # TODO this will raise an error if a block is empty
    block_neighbors = [[len(neighbors(graph, block_i[0], block_j))
                        for block_j in partition_as_list]
                       for block_i in partition_as_list]
    return vertices_per_block, block_neighbors


def are_common_partitions(graph1, partition1, graph2, partition2):
    # pre-condition: the partitions are valid and equitable
    #
    # `sizes` should be a list of length p and `block_neighbors` should be a
    # list of p lists, each of length p. The innermost entries should be
    # integers.
    sizes1, block_neighbors1 = partition_parameters(graph1, partition1)
    sizes2, block_neighbors2 = partition_parameters(graph2, partition2)
    combined1 = zip(sizes1, block_neighbors1)
    combined2 = zip(sizes2, block_neighbors2)
    # This is an inefficient algorithm, but it is quite readable.
    return any(perm == combined2 for perm in permutations(combined1))


def are_fractionally_isomorphic(G, H):
    """Returns ``True`` if and only if the graphs are fractionally isomorphic.

    """
    partition1 = coarsest_equitable_partition(G)
    partition2 = coarsest_equitable_partition(H)
    return are_common_partitions(G, partition1, H, partition2)


#: A graph consisting of a set of vertices and a set of edges.
#:
#: Each edge must be a :class:`frozenset` containing exactly two elements from
#: the set of vertices. For example, to create the circle graph on three
#: vertices::
#:
#:     vertices = {1, 2, 3}
#:     edges = {frozenset(1, 2), frozenset(2, 3), frozenset(3, 1)}
#:     G = Graph(vertices, edges)
#:
Graph = namedtuple('Graph', ['V', 'E'])
