# test_partitions.py - unit tests for fraciso.partitions
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
"""Unit tests for the :mod:`fraciso.partitions` module."""
from fraciso.partitions import coarsest_equitable_partition

from .helpers import graph_from_file


def test_coarsest_equitable_partition():
    G = graph_from_file('data/test_graph1.txt')
    partition = coarsest_equitable_partition(G)
    block1 = frozenset(range(6))
    block2 = frozenset(range(6, 12))
    assert frozenset((block1, block2)) == partition
