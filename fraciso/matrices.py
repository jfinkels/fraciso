# matrices.py - data structures and functions for square matrices
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
from itertools import permutations
from itertools import product

import numpy as np


def permutation_matrices(n):
    """Returns an iterator over all possible permutation matrices of size `n`.

    """
    return (np.mat(P) for P in permutations(np.identity(n)))


def _sequences_of_ones(n, k):
    """Returns a list of all binary sequences of length `n` with `k` ones."""
    if n <= 0:
        return [[]]
    if k <= 0:
        return [[0 for x in range(n)]]
    if k >= n:
        return [[1 for x in range(n)]]
    return [[0] + seq for seq in _sequences_of_ones(n - 1, k)] \
        + [[1] + seq for seq in _sequences_of_ones(n - 1, k - 1)]


def _matrices_with_row_sums(m, n, d):
    """Returns an iterator over all m by n matrices whose rows all sum to d.

    This function is dangerous! It creates a combinatorial explosion of
    matrices, so don't try to store all of the matrices in memory at once!

    """
    return product(_sequences_of_ones(n, d), repeat=m)


def _row_sum(matrix, m1, n1, m2, n2):
    """Computes the sum of the first row of the submatrix starting at (m1, n1)
    and ending at (m2, n2).

    `m2` must be greater than `m1` and `n2` must be greater than `n1`.

    """
    return sum(matrix[m1:m2, n1:n2][0])
