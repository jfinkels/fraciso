# __init__.py - indicates that this directory is a Python package
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

#: The current version of this extension.
#:
#: This should be the same as the version specified in the :file:`setup.py`
#: file.
__version__ = '0.0.7'

# The following names are part of the public API.
from fraciso.isomorphism import are_fractionally_isomorphic
#from fraciso.isomorphism import fractionally_isomorphic_graphs
from fraciso.isomorphism import random_fractionally_isomorphic_graph
from fraciso.isomorphism import random_fractionally_isomorphic_graphs
from fraciso.isomorphism import random_graph_from_parameters
from fraciso.isomorphism import verify_isomorphism
from fraciso.partitions import coarsest_equitable_partition
