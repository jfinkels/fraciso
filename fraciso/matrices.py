# matrices.py - functions for matrices
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
"""Functions that operate on matrices."""
from __future__ import division  # Just in case this is run on Python 2.X

from itertools import permutations
from itertools import product

import numpy as np

#: The data type for each entry in matrices created by functions in this
#: module.
DTYPE = np.int32


def permutation_matrices(n):
    """Returns an iterator over all possible permutation matrices of size `n`.

    """
    return (np.mat(P) for P in permutations(np.identity(n, dtype=DTYPE)))


def permutation_to_matrix(permutation):
    """Returns a permutation matrix corresponding to the permutation given by
    the specified dictionary.

    """
    p = [v for k, v in sorted(permutation.items())]
    return np.identity(len(p), dtype=DTYPE)[p, :]


def random_permutation_matrix(n, seed=None):
    """Returns a random `n` by `n` permutation matrix.

    If `seed` is specified, it is used to specify the random seed for the
    pseudorandom number generator.

    """
    np.random.seed(seed)
    p = np.random.permutation(n)
    # Only need to permute the rows, since this is the identity matrix.
    return np.identity(n, dtype=DTYPE)[p, :]


def is_doubly_stochastic(matrix):
    """Returns ``True`` if and only if the specified NumPy matrix is doubly
    stochastic.

    A matrix is **doubly stochastic** if the sum of the entries in each row is
    1 and the sum of the entries in each column is 1.

    """
    rows, columns = matrix.shape
    ones_m = np.mat(np.ones(columns)).T
    ones_n = np.mat(np.ones(rows)).T
    return (np.all(matrix * ones_n == ones_m)
            and np.all(ones_m.T * matrix == ones_n))


def dictionary_to_permutation(permutation):
    """Converts the specified permutation given as a dictionary to the
    corresponding permutation matrix.

    `permutation` must be a dictionary representing a permutation on the first
    **n** nonnegative integers, where **n** is the length of the dictionary.

    """
    n = len(permutation)
    matrix = np.mat(np.zeros((n, n)))
    for k, v in permutation.items():
        matrix[k, v] = 1.
        #matrix[v, k] = 1.
    return matrix


def _swap(obj, i, j):
    """Swaps the elements at index `i` and index `j` in `obj`.

    If `obj` has a ``copy()`` method, ``obj[j]`` gets a copy of ``obj[i]``.

    """
    temp = obj[i]
    if hasattr(temp, 'copy'):
        temp = temp.copy()
    obj[i] = obj[j]
    obj[j] = temp
    return obj


def _index_of_nonzero(column):
    """Returns the index of the first nonzero element in the specified array.

    """
    nonzero_entries = np.flatnonzero(np.asarray(column))
    if len(nonzero_entries) == 0:
        return -1
    return nonzero_entries[0]


def to_row_echelon(A):
    """Returns the row echelon form of `A` as computed by Gaussian elimination.

    This function returns a two-tuple. The first element is the row echelon
    form of A (possibly including permutations of the rows). The second element
    is the dictionary indicating the permutation applied to the matrix during
    the computation of the row echelons form. The dictionary includes a mapping
    from **i** to **j** if row **i** was swapped with row **j**.

    This function returns a new matrix; it does not modify `A`.

    """
    A = A.copy()
    rows, columns = A.shape
    # permutation stores the pivots that were performed: row i of the original
    # matrix was interchanged with row permutation[i]
    permutation = {x: x for x in range(rows)}
    # p is the current pivot column.
    p = 0
    # Iterate over each row (except the last one).
    for i in range(rows - 1):
        # If the entry at the current expected pivot point is zero, find the
        # next non-zero entry in this column and swap it.
        while A[i, p] == 0:
            nonzero = _index_of_nonzero(A[i:, p])
            # If there is some non-zero element, swap its row with the current
            # row and break out of this loop to move on to the elimination.
            if nonzero != -1:
                # nonzero is the index relative to row i, so we add nonzero to
                # i to determine the absolute row number in the matrix A.
                _swap(A, i, i + nonzero)
                _swap(permutation, i, i + nonzero)
                break
            # Otherwise, all elements in the current pivot column are zeros,
            # so increment the pivot column to try and find a usable column.
            p += 1
            # If we have exhausted all possible columns, this means the rest of
            # the matrix is zeros.
            if p == columns:
                return A, permutation
        # TODO this could be done more succinctly using a list comprehension
        #
        # Iterate over each row below the current one.
        for j in range(i + 1, rows):
            # k is the scalar by which we multiply the row before adding it.
            k = -A[j, p] / A[i, p]
            A[j] += k * A[i]
        # Increment the pivot column when we move to the next row.
        p += 1
        if p == columns:
            return A, permutation
    return A, permutation


def _sequences_of_ones(n, k):
    """Returns a list of all binary sequences of length `n` with `k` ones."""
    # There is exactly one sequence of length zero: the empty sequence.
    if n <= 0:
        return [[]]
    # There is exactly one sequence with no ones: the all-zeros sequence.
    if k <= 0:
        return [[0 for x in range(n)]]
    # There is exactly one sequence with all ones: the all-ones sequence.
    if k >= n:
        return [[1 for x in range(n)]]
    # In the recursive case, there are two possibilities: prepend a 0 to the
    # sequence or prepend a 1 to the sequence.
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
