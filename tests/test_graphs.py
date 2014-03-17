# test_graphs.py - unit tests for fraciso.graphs
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
"""Unit tests for the :mod:`fraciso.graphs` module."""
from fraciso.graphs import graph_from_file
from fraciso.graphs import neighbors

from tests.helpers import path_to


def test_neighbors():
    G = graph_from_file(path_to('data/test_graph1.txt'))
    assert neighbors(G, 0) == {1, 2, 3, 6}
    assert neighbors(G, 0, set(range(6))) == {1, 2, 3}
    assert neighbors(G, 0, set(range(6, 12))) == {6}
