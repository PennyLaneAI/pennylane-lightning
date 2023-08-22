Lightning
#########

:Release: |release|

.. image:: _static/pennylane_lightning.png
    :align: left
    :width: 200px
    :target: javascript:void(0);

.. include:: ../README.rst
  :start-after:   header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove


Devices
-------

Lightning provides the following devices:

.. title-card::
    :name: 'lightning.qubit'
    :description: A fast state-vector qubit simulator written in C++
    :link: lightning_qubit/device.html

.. title-card::
    :name: 'lightning.kokkos'
    :description: A heterogeneous backend state-vector simulator with Kokkos library support.
    :link: lightning_kokkos/device.html

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   lightning_qubit/device
   lightning_kokkos/device

.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   lightning_qubit/development/index

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
   C++ API <api/library_root>
