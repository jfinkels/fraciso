# linprog.py - interfaces to linear programming solvers
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
"""Interfaces to linear programming solvers for use in determining whether two
graphs are fractionally isomorphic.

"""
try:
    from cvxopt import matrix as cvx_matrix
    from cvxopt import solvers as cvx_solvers
    cvxopt_available = True
except ImportError:
    cvxopt_available = False
try:
    import ecos
    from scipy.sparse import csc_matrix
    ecos_available = True
except ImportError:
    ecos_available = False
from networkx import to_numpy_matrix
import numpy as np
try:
    #from pulp import COIN
    from pulp import GLPK
    from pulp import LpMinimize
    from pulp import LpProblem
    from pulp import LpStatusOptimal
    from pulp import lpSum
    from pulp import LpVariable
    pulp_available = True
except ImportError:
    pulp_available = False

from fraciso.matrices import dictionary_to_permutation
from fraciso.matrices import to_row_echelon


#: Known linear programming backends.
LP_METHODS = ('ecos', 'cvxopt', 'pulp')


def _pulp_dot_product(costs, variables):
    """Returns the PuLP-specific "dot product" of the given iterables.

    `costs` is an iterable of numbers specifying the weight or cost of the
    corresponding variable.

    `variables` is an iterable of :class:`LpVariable` instances.

    `costs` and `variables` must have the same length.

    """
    return lpSum(a * b for a, b in zip(costs, variables))


def pulp_solver(G, h, A, b, c, n):
    # First, create a variable for each of the columsn of G and A.
    #
    # pre-condition: G and A have the same number of columns.
    #
    # The second argument specifies a lower bound for the variable, so we can
    # safely ignore the inequality constraints given by G and h.
    variables = [LpVariable('s{}'.format(i), 0) for i in range(G.shape[1])]
    # LpVariable has a second argument that allows you to specify a lower bound
    # for the variable (for example, x1 >= 0). We don't specify nonnegativity
    # here, because it is already specified by the inequality constraints G and
    # h.
    #variables = [LpVariable('s{}'.format(i)) for i in range(G.shape[1])]
    # Next, create a problem context object and add the objective function c to
    # it. The first object added to LpProblem is implicitly interpreted as the
    # objective function.
    problem = LpProblem('fraciso', LpMinimize)
    # The np.dot() function doesn't like mixing numbers and LpVariable objects,
    # so we compute the dot product ourselves.
    #
    #problem += np.dot(variables, c), 'Dummy objective function'
    problem += _pulp_dot_product(c, variables), 'Dummy objective function'
    # Add each equality constraint to the problem context.
    for i, (row, b_value) in enumerate(zip(A, b)):
        #problem += np.dot(row, variables), 'Constraint {}'.format(i)
        # Convert the row to a list so pulp has an easier time dealing with it.
        row_as_list = np.asarray(row).flatten().tolist()
        dot_product = _pulp_dot_product(row_as_list, variables)
        problem += dot_product == b_value, 'Constraint {}'.format(i)
    solver_backend = GLPK()
    #solver_backend = COIN()
    problem.solve(solver_backend)
    if problem.status == LpStatusOptimal:
        # PuLP is silly and sorts the variables by name before returning them,
        # so we need to re-sort them in numerical order.
        solution = [s.varValue for s in sorted(problem.variables(),
                                               key=lambda s: int(s.name[1:]))]
        return True, solution
    # TODO status could be unknown here, but we're currently ignoring that
    return False, None


def ecos_solver(G, h, A, b, c, n):
    # In order to use ecos, we must provide sparse matrices
    A = csc_matrix(A)
    G = csc_matrix(G)
    dims = dict(l=n ** 2, q=[])
    solution = ecos.solve(c, G, h, dims, A, b)
    if solution['info']['exitFlag'] == 0:
        return True, solution['x']
    # TODO status could be unknown here, but we're currently ignoring that
    return False, None


def cvxopt_solver(G, h, A, b, c, n):
    # cvxopt doesn't allow redundant constraints in the linear program Ax = b,
    # so we need to do some preprocessing to find and remove any linearly
    # dependent rows in the augmented matrix [A | b].
    #
    # First we do Gaussian elimination to put the augmented matrix into row
    # echelon (reduced row echelon form is not necessary). Since b comes as a
    # numpy array (that is, a row vector), we need to convert it to a numpy
    # matrix before transposing it (that is, to a column vector).
    b = np.mat(b).T
    A_b = np.hstack((A, b))
    A_b, permutation = to_row_echelon(np.hstack((A, b)))
    # Next, we apply the inverse of the permutation applied to compute the row
    # echelon form.
    P = dictionary_to_permutation(permutation)
    A_b = P.I * A_b
    # Trim any rows that are all zeros. The call to np.any returns an array of
    # Booleans that correspond to whether a row in A_b is all zeros. Indexing
    # A_b by an array of Booleans acts as a selector. We need to use np.asarray
    # in order for indexing to work, since it expects a row vector instead of a
    # column vector.
    A_b = A_b[np.any(np.asarray(A_b) != 0, axis=1)]
    # Split the augmented matrix back into a matrix and a vector.
    A, b = A_b[:, :-1], A_b[:, -1]
    # Apply the linear programming solver; cvxopt requires that these are all
    # of a special type of cvx-specific matrix.
    G, h, A, b, c = (cvx_matrix(M) for M in (G, h, A, b, c))
    solution = cvx_solvers.lp(c, G, h, A, b)
    if solution['status'] == 'optimal':
        return True, solution['x']
    # TODO status could be 'unknown' here, but we're currently ignoring that
    return False, None


def fraciso_using_lp(graph1, graph2, method='ecos'):
    """Solves the fractional graph isomorphism problem by solving the
    corresponding linear program.

    `graph1` and `graph2` are instances of :class:`networkx.Graph`.

    `method` is a string indicating which linear programming library to use.
    Possible values are ``'ecos'``_ (the default), ``'cvxopt'``_, and
    ``'pulp'``_ (see also :data:`LP_METHODS`).

    If M and N are the adjacency matrices of the two graphs, then deciding if G
    is fractionally isomorphic to H is equivalent to deciding if the following
    linear program is feasible.

        M S = S N
        S 1 = 1
        1^T S = 1
        S >= 0

    Matrices separated by a single space denotes matrix multiplication. The
    ``1`` denotes the all ones vector. The first equality requires that there
    is a fuzzy alignment between the vertices of G and the vertices of H. The
    second requires that S is row-stochastic. The third requires that S is
    column-stochastic. The last is a shorthand for component-wise inequality,
    and requires that S is real but non-negative.

    This can be translated into a more standardized form of linear programming
    constraints of the form

        G x <= h
        A x  = b

    by setting the variables to be

        x = [s_11, s_12, ..., s_1n, s_21, ..., s_2n, ..., s_n1, ...s_nn]^T

    and setting the matrices as follows. (Below, ``(x)`` denotes the tensor
    product, also known as the "Kronecker product" [this is how it is known to
    NumPy].)

        G = -I (of dimension n^2 by n^2)
        h = 0  (of dimension n^2 by 1)
            __                       __
        A = |       (I (x) 1)^T       |  (of dimension 2n + n^2 by n^2)
            |       (1 (x) I)^T       |
            |_(M (x) I) - (I (x) N^T)_|
            __ __
        b = | 1 |  (of dimension 2n + n^2 by 1)
            | 1 |
            |_0_|

    Since we require an objective function in order to have a well-defined
    instance of the linear programming problem, we choose an arbitrary
    objective function, say the all ones vector, since we care only about
    feasibility, not about optimality.

    .. _ecos: https://github.com/ifa-ethz/ecos
    .. _cvxopt: http://cvxopt.org/
    .. _pulp: https://code.google.com/p/pulp-or

    """
    # Sanity check
    #assert len(G) == len(H)
    # For the sake of brevity, let I(n) be the identity matrix of dimension n.
    I = np.identity
    # Get the number of vertices in the graph.
    n = len(graph1)
    # Let M and N be the adjacency matrices of G and H respectively.
    M = to_numpy_matrix(graph1)
    N = to_numpy_matrix(graph2)
    # Set G and h, the linear inequalities part of the linear program. We use
    # these only to require that each entry of S must be positive.
    G = -I(n ** 2)
    h = np.zeros(n ** 2)
    # Set A and b, the linear equalities part of the linear program. We use
    # these to require three things. First, that S is row-stochastic. Second,
    # that S is column-stochastic. Third, that M S = S N, that is, that there
    # is a fuzzy correlation between blocks of vertices in the graphs.
    #
    # To reflect these three requirements, A and b are both defined in three
    # blocks, as shown in the docstring. In NumPy, they would be defined as
    # follows.
    A_1 = np.kron(I(n), np.ones(n))
    A_2 = np.kron(np.ones(n), I(n))
    A_3 = np.kron(M, I(n)) - np.kron(I(n), N.T)
    A = np.vstack((A_1, A_2, A_3))
    b_1 = np.ones(n)
    b_2 = np.ones(n)
    b_3 = np.zeros(n ** 2)
    b = np.hstack((b_1, b_2, b_3))
    # Use a bogus objective function, since we care only if a feasible solution
    # to the linear program exists.
    c = np.ones(n ** 2)
    if method == 'ecos':
        solver = ecos_solver
    elif method == 'cvxopt':
        solver = cvxopt_solver
    elif method == 'pulp':
        solver = pulp_solver
    else:
        raise ValueError('Unknown linear programming method:'
                         ' {}'.format(method))
    is_feasible, solution = solver(G, h, A, b, c, n)
    # Convert the solution vector into the shape of the doubly stochastic
    # matrix S that witnesses the equality MS = SN, where M and N are the
    # adjacency matrices of the two graphs.
    S = np.array(solution).reshape(n, n) if is_feasible else None
    return is_feasible, S
