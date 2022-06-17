========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/bregman-learning/badge/?style=flat
    :target: https://bregman-learning.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/TJHeeringa/bregman-learning/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/TJHeeringa/bregman-learning/actions

.. |requires| image:: https://requires.io/github/TJHeeringa/bregman-learning/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/TJHeeringa/bregman-learning/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/TJHeeringa/bregman-learning/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/TJHeeringa/bregman-learning

.. |version| image:: https://img.shields.io/pypi/v/bregman-learning.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/bregman-learning

.. |wheel| image:: https://img.shields.io/pypi/wheel/bregman-learning.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/bregman-learning

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/bregman-learning.svg
    :alt: Supported versions
    :target: https://pypi.org/project/bregman-learning

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/bregman-learning.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/bregman-learning

.. |commits-since| image:: https://img.shields.io/github/commits-since/TJHeeringa/bregman-learning/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/TJHeeringa/bregman-learning/compare/v0.0.0...main



.. end-badges

A pytorch extension providing the Bregman optimizers

* Free software: BSD 3-Clause License

Installation
============

::

    pip install bregman-learning

You can also install the in-development version with::

    pip install https://github.com/TJHeeringa/bregman-learning/archive/main.zip


Documentation
=============


https://bregman-learning.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
