Algorithms for fractional graph isomorphism
===========================================

This package contains algorithms for deciding if two graphs are fractionally
isomorphic. It requires Python 3.

This file was last updated on March 19, 2014.

Copyright license
-----------------

This package is distributed under the terms of the GNU General Public License
version 3. For more information, see the file `LICENSE` in this directory.

Basic usage
-----------

    from networkx import Graph
    from fraciso import are_fractionally_isomorphic

    # Create a NetworkX Graph objects G and H.
    #G = ...
    #H = ...
    
    print('G is fractionally isom. to H?', are_fractionally_isomorphic(G, H))

Installation requirements
-------------------------

    pip install -r requirements.txt

Testing
-------

    pip install -r requirements-test.txt
    nosetests

Contact
-------

Jeffrey Finkelstein <jeffrey.finkelstein@gmail.com>
