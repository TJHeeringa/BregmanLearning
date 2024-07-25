========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
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

A pytorch extension providing Bregman-based optimizers

* Free software: BSD 3-Clause License

Installation
============

The package can be install from PyPI using::

    pip install bregman-learning


Usage
============

The library provides 2 Bregman-based optimizers, several regularizers for these optimizers, and functions for pre- and postprocessing the networks.

The Bregman-based optimizers provides are LinBreg and AdaBreg. Their usage is similar to the usage of Adam and SGD, their non-Bregman counterparts. Instead of::

    from torch.optim import Adam

    ...

    optimizer = Adam(model.parameters(), lr=learning_rate)

the optimizers are created using::

    from bregman import AdaBreg, L1

    ...

    optimizer = AdaBreg(
        model.parameters(),
        reg=L1(rc=regularization_constant),
        lr=learning_rate
    )

where the L1 regularizer can be interchanged with any regularizer in the library.

For the best results when using sparsity-promoting regularizers, the networks have to pre- and postprocessed accordingly. For the L12 regularizer, this can be done using::

    from bregman import simplify, sparsify

    ...

    sparsify(model, density_level=0.2)

    ...

    pruned_model = simplify(model)


Citing
============
If you use this code, please use the citation information in the CITATION.cff file or click the `cite this repository` button in the sitebar.
