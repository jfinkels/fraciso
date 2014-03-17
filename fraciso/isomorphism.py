# isomorphism.py - algorithms for checking fractional isomorphism of graphs
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
"""Algorithms for determining if two graphs are fractionally isomorphic."""
from networkx.convert import from_numpy_matrix
from networkx.convert import to_numpy_matrix

from fraciso.partitions import are_common_partitions
from fraciso.partitions import coarsest_equitable_partition
from fraciso.partitions import partition_to_permutation


def are_fractionally_isomorphic(G, H):
    """Returns ``True`` if and only if the graphs are fractionally isomorphic.

    Two graphs are **fractionally isomorphic** if they share a common coarsest
    equitable partition.

    """
    partition1 = coarsest_equitable_partition(G)
    partition2 = coarsest_equitable_partition(H)
    return are_common_partitions(G, partition1, H, partition2)


def _enumerate(matrix, extents):
    # TODO could also use partition_parameters() to get row_sums
    #extents = [0] + extents
    #pairs = list(zip(extents, extents[1:]))
    pass
    #### Other attempts below
    # # For the sake of brevity, rename this function.
    # #
    # # This produces an iterator over all matrices in the same shape as the
    # # block of `matrix` starting at (m1, n1) and ending at (m2, n2), and with
    # # the same row sum as each row in that submatrix.
    # mx = lambda m1, n1, m2, n2: \
    #     _matrices_with_row_sums(m2 - m1, n2 - n1,
    #                             _row_sum(matrix, m1, n1, m2, n2))
    #######
    # # Mapping from submatrix boundaries to iterator over all possible
    # # submatrices with the same row sums.
    # b = {(m1, m2): {(n1, n2): mx(m1, n1, m2, n2) for n1, n2 in pairs}
    #      for m1, m2 in pairs}
    # # Mapping from submatrix row boundaries to list of all possible rows.
    # rows = {(m1, m2): [reduce(_append_matrices, row)
    #                    for row in product(*
    #                      lexicographic_blocks(b[(m1, m2)]))]
    #         for m1, m2 in pairs}
    # # Iterator over all full matrices.
    # full = [sum(r) for r in product(*lexicographic_blocks(rows))]
    # return [Matrix(m) for m in full]
    ######
    # # THIS IS TOOO COMPLICATED
    # return (sum(rows)
    #         for rows in product(*((
    #                 reduce(_append_matrices, row_of_blocks)
    #                 for row_of_blocks in product(*(
    #                         mx(m1, n1, m2, n2)
    #                         for n1, n2 in pairs)))
    #                               for m1, m2 in pairs)))


def fractionally_isomorphic_graphs(graph):
    """Returns an iterator over all graphs that are fractionally isomorphic to
    the specified graph, including the graph itself.

    """
    partition = coarsest_equitable_partition(graph)
    permutation, extents = partition_to_permutation(graph, partition)
    # TODO this would be simpler if we used an adjacency matrix representation
    block_matrix = permutation * to_numpy_matrix(graph)
    # At this point, we have the block matrix and the "extents" (the indices
    # giving the bounds of the blocks of the matrix). Now we enumerate each
    # possible submatrix with the same row sum.
    raise NotImplementedError
    return (from_numpy_matrix(M) for M in _enumerate(block_matrix, extents))
