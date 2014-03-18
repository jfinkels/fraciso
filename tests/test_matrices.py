# test_matrices.py - unit tests for fraciso.matrices
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
"""Unit tests for the :mod:`fraciso.matrices` module."""
from __future__ import division

import numpy as np

from fraciso.matrices import dictionary_to_permutation
from fraciso.matrices import _matrices_with_row_sums
from fraciso.matrices import _sequences_of_ones
from fraciso.matrices import to_row_echelon


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


def test_dictionary_to_permutation():
    permutation = {0: 1, 1: 2, 2: 0}
    expected = np.mat([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    actual = dictionary_to_permutation(permutation)
    assert np.all(expected == actual)


def test_row_echelon():
    A = np.array([[2., 1, -1, 8],
                  [-3, -1, 2, -11],
                  [-2, 1, 2, -3]])
    R, p = to_row_echelon(A)
    assert np.all(R[:, 0] == np.array([2, 0, 0]))
    assert np.all(R[1:, 1] == np.array([1 / 2, 0]))
    assert np.all(R[2:, 2] == np.array([-1]))

    A = np.array([[0., 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]])
    R, p = to_row_echelon(A)
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    assert np.all(R == expected)

    A = np.array([[1., 0, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 1, 1]])
    R, p = to_row_echelon(A)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]])
    assert np.all(R == expected)

    A = np.array([[1., 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 2]])
    R, p = to_row_echelon(A)
    expected = np.array([[1, 1, 1, 1],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]])
    assert np.all(R == expected)
