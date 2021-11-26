PennyLane-Lightning Plugin
##########################

.. image:: https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane-lightning/Testing/master?logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-lightning/actions?query=workflow%3ATesting

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-lightning/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-lightning

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-lightning/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-lightning

.. image:: https://img.shields.io/readthedocs/pennylane-lightning.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylane-lightning.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-Lightning

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-Lightning.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-Lightning

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Lightning plugin provides a fast state-vector simulator written in C++.

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

PennyLane-Lightning requires Python version 3.7 and above. It can be installed using ``pip``:

.. code-block:: console

    $ pip install pennylane-lightning

To build PennyLane-Lightning from source you can run

.. code-block:: console

    $ pip install pybind11 pennylane-lightning --no-binary :all:

A C++ compiler such as ``g++``, ``clang``, or ``MSVC`` is required. On Debian-based systems, this
can be installed via ``apt``:

.. code-block:: console

    $ sudo apt install g++

The `pybind11 <https://pybind11.readthedocs.io/en/stable/>`_ library is also used for binding the
C++ functionality to Python.

Alternatively, for development and testing, you can install by cloning the repository:

.. code-block:: console

    $ git clone https://github.com/XanaduAI/pennylane-lightning.git
    $ cd pennylane-lightning
    $ pip install -r requirements.txt
    $ pip install -e .

Note that subsequent calls to ``pip install -e .`` will use cached binaries stored in the
``build`` folder. Run ``make clean`` if you would like to recompile.

You can also pass ``cmake`` options with ``build_ext``:

.. code-block:: console

    $ python3 setup.py build_ext -i --define="ENABLE_OPENMP=OFF;ENABLE_NATIVE=ON"

and install the compilied library with

.. code-block:: console

    $ python3 setup.py develop

Testing
-------

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

.. code-block:: console

    $ make test-python

while the C++ code can be tested with

.. code-block:: console

    $ make test-cpp


CMake Support
-------------

One can also build the plugin using CMake:

.. code-block:: console

    $ cmake -S. -B build
    $ cmake --build build

To test the C++ code:

.. code-block:: console

    $ mkdir build && cd build
    $ cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
    $ make

Other supported options are ``-DENABLE_WARNINGS=ON``,
``-DENABLE_NATIVE=ON`` (for ``-march=native``), 
``-DENALBE_OPENMP=ON``, ``-DENALBE_BLAS=ON``, and
``-DENABLE_CLANG_TIDY=ON``.



Compile on Windows with MSVC
----------------------------

You can also compile Pennylane-Lightning on Windows using `Microsoft Visual C++ <https://visualstudio.microsoft.com/vs/features/cplusplus/>`_ compiler. You need `cmake <https://cmake.org/download/>`_ and appropriate Python environment (e.g. using `Anaconda <https://www.anaconda.com/>`_).


We recommend to use ``[x64 (or x86)] Native Tools Command Prompt for VS [version]`` for compiling the library. Be sure that ``cmake`` and ``python`` can be called within the prompt.


.. code-block:: console

    $ cmake --version
    $ python --version

Then a common command will work.

.. code-block:: console

    $ pip install -r requirements.txt
    $ pip install -e .

Note that OpenMP and BLAS are disabled in this setting.


.. installation-end-inclusion-marker-do-not-remove


Please refer to the `plugin documentation <https://pennylane-lightning.readthedocs.io/>`_ as
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

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning/issues
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
