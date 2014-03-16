Algorithms for fractional graph isomorphism
===========================================

This package contains algorithms for deciding if two graphs are fractionally
isomorphic.

This file was last updated on March 14, 2014.

Copyright license
-----------------

This package is distributed under the terms of the GNU General Public License
version 3. For more information, see the file `LICENSE` in this directory.

Requirements
------------

This package requires Python version 3 or later.

Basic usage
-----------

    from fractionalisomorphism import Graph
    from fractionalisomorphism import are_fractionally_isomorphic

    # Create a graph G.
    vertices = {1, 2, 3}
    edges = {frozenset(1, 2), frozenset(2, 3), frozenset(3, 1)}
    G = Graph(vertices, edges)

    # Create a graph H.
    #H = ...

    # Determine if they are fractionally isomorphic.
    if are_fractionally_isomorphic(G, H):
        print('Yes')
    else:
        print('No')

Testing
-------

Running the unit tests requires [nose][0].

    pip install nose

Run the tests as follows.

    nosetests

[n]: https://nose.readthedocs.org/

Contact
-------

Jeffrey Finkelstein <jeffrey.finkelstein@gmail.com>
