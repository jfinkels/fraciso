.. fraciso documentation master file, created by
   sphinx-quickstart on Mon Mar 31 19:03:33 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Algorithms for fractional isomorphism
=====================================

This package provides functions that decide whether two graphs are fractionally
isomorphic, as well as functions that generate graphs fractionally isomorphic
to a given initial graph.

`Source code (github.com) <https://github.com/jfinkels/fraciso>`_ ·
`Packaging (pypi.python.org) <https://pypi.python.org/pypi/fraciso>`_ ·
`Issues (github.com) <https://github.com/jfinkels/fraciso/issues>`_

Installation
------------

.. sourcecode:: bash

   pip install fraciso

Basic usage
-----------

.. sourcecode:: python

   from fraciso import are_fractionally_isomorphic
   from fraciso import coarsest_equitable_partition
   from fraciso import random_fractionally_isomorphic_graph
   from networkx import Graph

   # Create some NetworkX graphs.
   G = Graph()
   H = Graph()

   #
   # Add some vertices and edges to graphs G and H here...
   #

   # Decide whether two graphs are fractionally isomorphic.
   print('Isomorphic?', are_fractionally_isomorphic(G, H))

   # Use linear programming to decide whether they are fractionally isomorphic.
   #
   # This requires PuLP and GLPK to be installed.
   print('Isomorphic?', are_fractionally_isomorphic(G, H, algorithm='lp.pulp'))

   # Generate a random graph that is fractionally isomorphic to G.
   J = random_fractionally_isomorphic_graph(G)

   # Compute the coarsest equitable partition of a graph.
   P = coarsest_equitable_partition(G)


Mathematical background
-----------------------

An undirected graph is a pair :math:`(V, E)`, where :math:`V` is a set of
vertices and :math:`E` is a subset of :math:`V \times V`. If we assume the
graph has :math:`n` vertices and :math:`V = \{0, \dotsc, n - 1\}`, then the
*adjacency matrix* of a graph is the matrix :math:`A` in which entry
:math:`A_{ij}` is :math:`1` if :math:`i` is adjacent to :math:`j` and :math:`0`
otherwise.

Graph isomorphism
~~~~~~~~~~~~~~~~~

Suppose :math:`A` and :math:`B` are the adjacency matrices of graphs :math:`G`
and :math:`H`, respectively. The graphs :math:`G` and :math:`H` are
*isomorphic* if the following linear program has a feasible solution.

.. math::

   AP & = PB \\
   P 1 & = 1 \\
   1^T P & = 1 \\
   P & \in \{0, 1\}^{n \times n}


In other words, there is a permutation of the vertices of :math:`G` that makes
the adjacency matrices identical (any feasible :math:`P` must be a permutation
matrix).

Fractional graph isomorphism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The problem of determining whether two graphs are isomorphic is a
computationally difficult problem. However, a relaxation of the graph
isomorphism problem is computationally tractable. The *fractional graph
isomorphism* problem is the problem of determining whether the following linear
program has a feasible solution.

.. math::

   AS & = SB \\
   S 1 & = 1 \\
   1^T S & = 1 \\
   S & \geq 0

The final inequality is an abbreviation for the requirement that all entries of
:math:`S` must be real and nonnegative (so any feasible :math:`S` must be a
doubly stochastic matrix). This sole relaxation permits the problem to be
solved in polynomial time (by any efficient linear programming algorithm).

Algorithms
----------

Describe algorithms here.

Coarsest equitable partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Describe coarsest equitable partition algorithm here.

Linear programming
~~~~~~~~~~~~~~~~~~

Desrcibe linear programming algorithms here.

API
---

.. module:: fraciso

.. autofunction:: are_fractionally_isomorphic

.. autofunction:: coarsest_equitable_partition

.. autofunction:: random_fractionally_isomorphic_graph

.. autofunction:: random_fractionally_isomorphic_graphs

.. autofunction:: verify_isomorphism
