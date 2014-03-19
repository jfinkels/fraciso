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
from fraciso.matrices import _sequences_of_ones
from fraciso.matrices import _matrices_with_row_sums


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
