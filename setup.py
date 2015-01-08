# setup.py - installation script for this package
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
"""Algorithms for fractional graph isomorphism.

* `Documentation <http://readthedocs.org/docs/fraciso>`_
* `Packaging <http://pypi.python.org/pypi/fraciso>`_
* `Source code <http://github.com/jfinkels/fraciso>`_
* `Issues <http://github.com/jfinkels/fraciso/issues>`_

"""
from setuptools import setup

#: Installation requirements.
requirements = ['networkx>=1.9.1', 'numpy', 'scipy']


setup(
    author='Jeffrey Finkelstein',
    author_email='jeffrey.finkelstein@gmail.com',
    #classifiers=[],
    description='Algorithms for fractional graph isomorphism',
    download_url='https://github.com/jfinkels/fraciso',
    install_requires=requirements,
    include_package_data=True,
    #keywords=[],
    license='GNU GPLv3+',
    long_description=__doc__,
    name='fraciso',
    platforms='any',
    packages=['fraciso'],
    test_suite='nose.collector',
    tests_require=['nose'],
    url='https://github.com/jfinkels/fraciso',
    version='0.0.7',
    zip_safe=False
)
