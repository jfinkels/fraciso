# test_isomorphism.py - unit tests for fraciso.isomorphism
#
# Copyright 2014 Jeffrey Finkelstein.
#
# This file is part of fractionalisomorphism.
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
"""Unit tests for the :mod:`fraciso.isomorphism` module."""
try:
    import cvxopt
    cvxopt_available = True
except ImportError:
    cvxopt_available = False
try:
    import ecos
    ecos_available = True
except ImportError:
    ecos_available = False
try:
    import pulp
    pulp_available = True
except ImportError:
    pulp_available = False

from fraciso.isomorphism import are_fractionally_isomorphic
from fraciso.isomorphism import random_fractionally_isomorphic_graph
from fraciso.isomorphism import random_fractionally_isomorphic_graphs
from fraciso.isomorphism import fractionally_isomorphic_graphs
from fraciso.isomorphism import verify_isomorphism
from fraciso.matrices import is_doubly_stochastic

from .helpers import graph_from_file
from .helpers import skip
from .helpers import skip_unless


def _assert_fractionally_isomorphic(G=None, H=None, algorithm='cep'):
    if G is None:
        G = graph_from_file('data/test_graph1.txt')
    if H is None:
        H = graph_from_file('data/test_graph2.txt')
    are_isomorphic, S = are_fractionally_isomorphic(G, H, algorithm=algorithm)
    assert are_isomorphic
    if algorithm != 'cep':
        assert verify_isomorphism(G, H, S)


def test_are_fractionally_isomorphic():
    # Test some graphs that are isomorphic, and therefore fractionally
    # isomorphic as well.
    G = graph_from_file('data/graph3.txt')
    H = graph_from_file('data/graph4.txt')
    _assert_fractionally_isomorphic(G, H)

    # Test some graphs that are fractionally isomorphic but not isomorphic.
    _assert_fractionally_isomorphic()


@skip('ecos gives an approximate solution')
@skip_unless(ecos_available, 'ecos not available.')
def test_ecos():
    _assert_fractionally_isomorphic(algorithm='lp.ecos')


@skip_unless(pulp_available, 'pulp not available.')
def test_pulp():
    _assert_fractionally_isomorphic(algorithm='lp.pulp')


@skip('cvxopt cannot solve linear programs that are not full rank')
@skip_unless(cvxopt_available, 'cvxopt not available.')
def test_cvxopt():
    _assert_fractionally_isomorphic(algorithm='lp.cvxopt')


@skip('fractionally_isomorphic_graphs() is not implemented')
def test_fractionally_isomorphic_graphs():
    G = graph_from_file('data/test_graph1.txt')
    H = graph_from_file('data/test_graph2.txt')
    isomorphic_graphs = list(fractionally_isomorphic_graphs(G))
    assert G in isomorphic_graphs
    assert H in isomorphic_graphs


def test_random_fractionally_isomorphic_graph():
    G = graph_from_file('data/test_graph1.txt')
    for seed in (123, 456, 789):
        H = random_fractionally_isomorphic_graph(G, seed=seed)
        assert are_fractionally_isomorphic(G, H)


def test_random_fractionally_isomorphic_graphs():
    G = graph_from_file('data/test_graph1.txt')
    seed = 123
    times = 3
    for H in random_fractionally_isomorphic_graphs(G, times=times, seed=seed):
        assert are_fractionally_isomorphic(G, H)
