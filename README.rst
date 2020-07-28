PennyLane-Lightning Plugin
##########################

.. image:: https://img.shields.io/codacy/grade/f4132f03ce224f82bd3e8ba436b52af3.svg?style=popout-square
    :alt: Codacy grade
    :target: https://www.codacy.com/app/XanaduAI/pennylane-lightning

|

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Lightning plugin provides a fast state-vector simulator written in C++ using `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_.

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove

Features
========

* Combine PennyLane-Lightning's high performance simulator with PennyLane's
  automatic differentiation and optimization.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

TODO

This plugin requires Python version 3.5 and above.  Installation of this
plugin, as well as all dependencies, can be done using ``pip``:

.. code-block:: bash

    pip install pennylane-lightning

To test that the PennyLane-Lightning plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

.. installation-end-inclusion-marker-do-not-remove

Please refer to the `plugin documentation <https://pennylanelightning.readthedocs.io/>`_ as
well as to the `PennyLane documentation <https://pennylane.readthedocs.io/>`_ for further reference.

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Authors
=======

PennyLane-Lightning is the work of `many contributors <https://github.com/XanaduAI/pennylane-lightning/graphs/contributors>`_.

If you are doing research using PennyLane and PennyLane-Lightning, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove

Support
=======

TODO - update

- **Source Code:** https://github.com/XanaduAI/pennylane-lightning
- **Issue Tracker:** https://github.com/XanaduAI/pennylane-lightning/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane lightning plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

.. license-end-inclusion-marker-do-not-remove
