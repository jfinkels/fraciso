# isomorphism.py - algorithms for checking fractional isomorphism of graphs
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
"""Algorithms for determining if two graphs are fractionally isomorphic."""
import itertools

from networkx import get_node_attributes
from networkx import Graph
from networkx import from_numpy_matrix
from networkx import to_numpy_matrix
from networkx import bipartite_configuration_model
from networkx import empty_graph
from networkx import random_regular_graph
import numpy as np
from scipy.linalg import block_diag

from fraciso.linprog import cvxopt_available
from fraciso.linprog import ecos_available
from fraciso.linprog import fraciso_using_lp
from fraciso.linprog import LP_METHODS
from fraciso.linprog import pulp_available
from fraciso.matrices import dictionary_to_permutation
from fraciso.matrices import is_doubly_stochastic
from fraciso.matrices import random_permutation_matrix
from fraciso.matrices import permutation_to_matrix
from fraciso.matrices import to_row_echelon
from fraciso.partitions import are_common_partitions
from fraciso.partitions import coarsest_equitable_partition
from fraciso.partitions import partition_parameters


def _cep_solver(G, H):
    """Solves the fractional graph isomorphism problem by comparing the
    coarsest equitable partitions of G and H.

    Returns a two-tuple whose left element is a Boolean representing whether
    the two graphs have a common coarsest partition and whose right element is
    the pair of partitions of the given graphs (in the same order as the inputs
    are given). If the left element is ``False``, the right element is
    ``None``, since there is no witness that the graphs are fractionally
    isomorphic.

    """
    partition1 = coarsest_equitable_partition(G)
    partition2 = coarsest_equitable_partition(H)
    result = are_common_partitions(G, partition1, H, partition2)
    witness = (partition1, partition2) if result else None
    return result, witness


def are_fractionally_isomorphic(G, H, algorithm='cep'):
    """Solves the fractional graph isomorphism problem and provides a witness,
    if any, that the two graphs are fractionally isomorphism.

    Two graphs are **fractionally isomorphic** if they share a common coarsest
    equitable partition. Equivalently, graphs with adjacency matrices A and B
    are fractionally isomorphic if there is a doubly stochastic matrix such
    that AS = SB.

    `G` and `H` must be instances of :class:`networkx.Graph`.

    `algorithm` can be either ``'cep'`` (the default) or ``'lp.LPSOLVER'``,
    where ``LPSOLVER`` is one of the recognized linear programming solvers
    (``ecos``, ``cvxopt``, etc.). The former denotes the coarsest equitable
    partition algorithm, which finds the coarsest equitable partition in either
    graph and checks that they are equivalent. The latter denotes one of the
    linear programming algorithms, which solves the linear programming
    formulation of the fractional graph isomorphism problem as specified above.

    The available linear programming solvers can be found in
    :data:`fraciso.linprog.LP_METHODS`.

    This function returns a two-tuple. The left element is a Boolean
    representing whether the two graphs are fractionally isomorphic. The right
    element is a witness that the two graphs are fractionally isomorphic, or
    ``None`` if no such witness exists. The witness depends on the algorithm

    If a specified linear programming library is known but unavailable, this
    function raises a :exc:`RuntimeError`. If an unknown algorithm is
    specified, this function raises a :exc:`ValueError`.

    """
    if algorithm == 'cep':
        return _cep_solver(G, H)
    if algorithm.startswith('lp'):
        try:
            method = algorithm.split('.')[1]
        except:
            raise ValueError('Must provide algorithm like "lp.ecos" or'
                             ' "lp.cvxopt"; got "{}"'.format(algorithm))
        if method not in LP_METHODS:
            raise ValueError('Unknown linear programming method'
                             ' {}'.format(method))
        if method == 'ecos' and not ecos_available:
            raise RuntimeError('ecos unavailable; try "pip install ecos"')
        if method == 'cvxopt' and not cvxopt_available:
            raise RuntimeError('cvxopt unavailable; try "pip install cvxopt"')
        if method == 'pulp' and not pulp_available:
            raise RuntimeError('pulp unavailable; try "pip install pulp"')
        return fraciso_using_lp(G, H, method)
    raise ValueError('Unknown algorithm: {}'.format(algorithm))


def _random_biregular_graph(num_left_vertices, num_right_vertices, left_degree,
                            right_degree, half_only=False, seed=None):
    """Returns the adjacency matrix of a random biregular graph with the
    specified number of left and right vertices, and the specified left and
    right degree.

    The returned adjacency matrix is a NumPy array (not a list of lists, not a
    NumPy matrix). If **L** and **R** are the number of left and right vertices
    respectively and **d** and **e** are the left and right degree
    respectively, then the returned adjacency matrix has the form::

         _      _
        |  0   B |
        |_B^T  0_|

    where B is an L by R matrix in which each row sums to **d** and each column
    sums to **e**. In this form, the first L nonnegative integers represent the
    left vertices and the following R integers represent the right vertices.

    If `half_only` is ``True``, this function returns only the submatrix B.

    If `seed` is specified, it must be an integer provided as the seed to the
    pseudorandom number generator used to generate the graph.

    .. note::

       Although Networkx includes a function for generating random bipartite
       graphs
       (:func:`networkx.generators.bipartite.bipartite_configuration_model`),
       their adjacency matrices do not necessarily have this block form.

    """
    # Rename some variables for brevity.
    L, R = num_left_vertices, num_right_vertices
    n = L + R
    # Generate a random graph with the appropriate degree sequence.
    left_sequence = [left_degree for x in range(L)]
    right_sequence = [right_degree for x in range(R)]
    # Need to use `create_using=Graph()` or else networkx will create a
    # multigraph.
    graph = bipartite_configuration_model(left_sequence, right_sequence,
                                          create_using=Graph(), seed=seed)
    # Find the nodes that are in the left and right sets. The `bipartite`
    # attribute specifies which set each vertex is in.
    left_or_right = get_node_attributes(graph, 'bipartite')
    left_nodes = (v for v, side in left_or_right.items() if side == 0)
    right_nodes = (v for v, side in left_or_right.items() if side == 1)
    all_nodes = itertools.chain(left_nodes, right_nodes)
    # Determine the permutation that moves all the left nodes to the top rows
    # of the matrix and all the right nodes to the bottom rows of the
    # matrix. The permutation maps row number to vertex that should be in that
    # row.
    permutation = {i: v for i, v in enumerate(all_nodes)}
    P = permutation_to_matrix(permutation)
    # Apply the permutation matrix to the adjacency matrix.
    M = to_numpy_matrix(graph)
    result = P * M
    # If `half_only` is True, only return the submatrix block consisting of the
    # first L rows and the last R columns.
    return result[:L, -R:] if half_only else result


def random_graph_from_parameters(vertices_per_block, block_neighbors,
                                 seed=None):
    """Returns a random graph that satisfies the given parameters.

    `vertices_per_block` and `block_neighbors` are the matrices returned by
    :func:`~fraciso.partitions.partition_parameters`.

    If `seed` is specified, it must be an integer provided as the seed to the
    pseudorandom number generator used to generate the graph.

    """
    # TODO there is an alternate way to implement this function: create a
    # random regular networkx.Graph object for each block of the partition,
    # create a random biregular Graph object between blocks of the partition,
    # then compute the union of the two graphs.
    #
    # Rename some variables for the sake of brevity.
    n, D = np.asarray(vertices_per_block), np.asarray(block_neighbors)
    # p is the number of blocks
    p = len(n)
    mat = to_numpy_matrix
    rr = lambda d, s: random_regular_graph(d, s, seed=seed)
    rb = lambda L, R, d, e:  _random_biregular_graph(L, R, d, e, True, seed)
    # Create a block diagonal matrix that has the regular graphs corresponding
    # to the blocks of the partition along its diagonal.
    regular_graphs = block_diag(*(mat(rr(d, s))
                                for s, d in zip(n, D.diagonal())))
    # Create a block strict upper triangular matrix containing the upper-right
    # blocks of the bipartite adjacency matrices.
    #
    # First, we create a list containing only the blocks necessary.
    blocks = [[rb(n[i], n[j], D[i, j], D[j, i]) for j in range(i + 1, p)]
              for i in range(p - 1)]
    # Next, we pad the lower triangular entries with blocks of zeros. (We also
    # need to add an extra block row of all zeros.) At this point, `padded` is
    # a square list of lists.
    padded = [[np.zeros((n[i], n[j])) for j in range(p - len(row))] + row
              for i, row in enumerate(blocks)]
    padded.append([np.zeros((n[-1], n[i])) for i in range(p)])
    # To get the block strict upper triangular matrix, we concatenate the block
    # matrices in each row.
    biregular_graphs = np.vstack(np.hstack(row) for row in padded)
    # Finally, we add the regular graphs on the diagonaly, the upper biregular
    # graphs, and the transpose of the upper biregular graphs in order to get a
    # graph that has the specified parameters.
    adjacency_matrix = regular_graphs + biregular_graphs + biregular_graphs.T
    return from_numpy_matrix(adjacency_matrix)


def random_fractionally_isomorphic_graph(graph, seed=None):
    """Returns a random graph that is fractionally isomorphic to the specified
    graph.

    This function may return the same graph it received as input. If the
    coarsest equitable partition of the input graph is the trivial partition
    (the partition in which each block contains exactly one vertex), then this
    function always returns the same graph it received as input.

    """
    # Get the parameters for the coarsest equitable partition of the graph; n
    # is the vector containing number of vertices in each block of the
    # partition and D_ij is the matrix containing the degree of any vertex in
    # block i relative to block j. p is the number of blocks in the partition.
    partition = coarsest_equitable_partition(graph)
    n, D = partition_parameters(graph, partition, as_matrices=True)
    p = len(n)
    # Permuting the parameters in an arbitrary fashion may produce parameters
    # for which there is no possible graph. For now, just stick with a random
    # graph on the same parameters.
    #
    ## Choose a random permutation matrix and permute the parameters.
    #P = random_permutation_matrix(p, seed)
    #n, D = P * n, P * D
    #
    # Get a random graph with those parameters.
    return random_graph_from_parameters(n, D, seed)


def random_fractionally_isomorphic_graphs(graph, times=None, seed=None):
    """Returns an iterator that generates random graphs that are fractionally
    isomorphic to the specified graph.

    If `times` is specified, the iterator only generates that number of graphs
    before it terminates. If it is not specified, it will generate graphs
    forever!

    If `seed` is specified, it must be an integer provided as the seed to the
    pseudorandom number generator used to generate the graph.

    This function may return the same graph.

    """
    # Get the parameters for the coarsest equitable partition of the graph; n
    # is the vector containing number of vertices in each block of the
    # partition and D_ij is the matrix containing the degree of any vertex in
    # block i relative to block j. p is the number of blocks in the partition.
    partition = coarsest_equitable_partition(graph)
    n, D = partition_parameters(graph, partition, as_matrices=True)
    p = len(n)
    # This code is almost exactly the same as itertools.repeat(), except that
    # we execute the random_graph_from_parameters() function every time
    # through the loop.
    if times is None:
        while True:
            # Permuting the parameters in an arbitrary fashion may produce
            # parameters for which there is no possible graph. For now, just
            # stick with a random graph on the same parameters.
            #
            ## Choose a random permutation matrix and permute the parameters.
            #P = random_permutation_matrix(p, seed)
            #n_prime, D_prime = P * n, P * D
            #
            # Get a random graph with those parameters.
            yield random_graph_from_parameters(n, D, seed)
    else:
        for i in range(times):
            # (See the note on permutations above.)
            #
            # Get a random graph with those parameters.
            yield random_graph_from_parameters(n, D, seed)


def fractionally_isomorphic_graphs(graph):
    """Returns an iterator over all graphs that are fractionally isomorphic to
    the specified graph, including the graph itself.

    `graph` must be an instance of :class:`networkx.Graph`.

    .. warning::

       This function is currently not implemented, since it is very difficult
       to write correct code that enumerates each graph that is fractionally
       isomorphic to a given graph (that is, it seems that the code will be
       very complicated).

    """
    raise NotImplementedError


def verify_isomorphism(G, H, S):
    """Returns ``True`` if and only if the doubly stochastic matrix `S` is a
    correct witness that `G` and `H` are fractionally isomorphic.

    `G` and `H` are instances of :class:`networkx.Graph`. `S` is a Numpy
    matrix.

    If **A** and **B** are the adjacency matrices of `G` and `H`, respectively,
    then this function checks that **AS = SB**, and that `S` is nonnegative and
    doubly stochastic.

    """
    A = to_numpy_matrix(G)
    B = to_numpy_matrix(H)
    return (np.all(S >= 0) and is_doubly_stochastic(S)
            and np.array_equal(A * S, S * B))
