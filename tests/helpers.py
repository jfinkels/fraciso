# helpers.py - helper functions for unit tests
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
"""Helper functions for unit tests."""
import os.path

from networkx.convert import from_numpy_matrix
from nose import SkipTest
from numpy import loadtxt

#: The absolute path to the directory containing this file.
DIRPATH = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def skip(reason=None):
    """Decorator that skips the decorated test function.

    This is a replacement for :func:`unittest.skip` that works with
    ``nose``. The argument ``reason`` is a string describing why the test was
    skipped.

    """
    def skipped(test):
        # If no reason is given, don't display one in the message.
        if reason:
            message = 'Skipped {0}: {1}'.format(test.__name__, reason)
        else:
            message = 'Skipped {0}'.format(test.__name__)

        # TODO Since we don't check the case in which `test` is a class, the
        # result of running the tests will be a single skipped test, although
        # it should show one skip for each test method within the class.
        def inner(*args, **kw):
            raise SkipTest(message)
        inner.__name__ = test.__name__
        return inner
    return skipped


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
