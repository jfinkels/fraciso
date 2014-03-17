# helpers.py - helper functions for unit tests
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
"""Helper functions for unit tests."""
import os.path

from networkx.convert import from_numpy_matrix
from numpy import loadtxt

#: The absolute path to the directory containing this file.
DIRPATH = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def path_to(filename):
    """Returns a path to the specified file, given as a path relative to the
    directory containing **this** file.

    """
    return os.path.join(DIRPATH, filename)


def graph_from_file(filename):
    """Reads an adjacency matrix from a file and returns an instance of
    :data:`Graph`.

    The adjacency matrix must have entries separated by a space and rows
    separated by a newline.

    """
    return from_numpy_matrix(loadtxt(path_to(filename)))
