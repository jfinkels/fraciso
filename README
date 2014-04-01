# Algorithms for fractional graph isomorphism #

This package contains algorithms for deciding if two graphs are fractionally
isomorphic. It requires Python 3.

This file was last updated on March 25, 2014.

## Copyright license ##

This package is distributed under the terms of the GNU General Public License
version 3. For more information, see the file `LICENSE` in this directory.

## Basic usage ##

    from networkx import Graph
    from fraciso import are_fractionally_isomorphic

    # Create a NetworkX Graph objects G and H.
    #G = ...
    #H = ...
    
    print('G is fractionally isom. to H?', are_fractionally_isomorphic(G, H))

### Using linear programming solvers ###

The fractional graph isomorphism problem can be solved via linear
programming. In order to use a supported linear programming solver, it must be
installed on the system (see installation requirements below). In order to
instruct this package to use a linear programming solver, use the ``algorithm``
keyword argument to the ``are_fractionally_isomorphic`` function. For example:

    print('G is fractionally isom. to H?',
          are_fractionally_isomorphic(G, H, algorithm='lp.ecos'))

## Installation requirements ##

In order to install the Python libraries required to use this package:

    pip install -r requirements.txt

If you want to use the PuLP library with the GLPK backend to solve the linear
programming formulation of the fractional graph isomorphism problem, install
GLPK:

      sudo aptitude install glpk-utils

If you want to use the PuLP library with the COIN-OR CBC backend to solve the
linear programming formulation of the fractional graph isomorphism problem,
install COIN-OR CBC:

      sudo aptitude install coinor-libcbc0

Testing
-------

    pip install -r requirements-test.txt
    nosetests

Release instructions
--------------------

This is a reminder for the maintainer of this package.

    python setup.py egg_info sdist upload --sign

Contact
-------

Jeffrey Finkelstein <jeffrey.finkelstein@gmail.com>
