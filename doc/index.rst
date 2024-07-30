Lightning plugins
#################

:Release: |release|

.. image:: _static/pennylane_lightning.png
    :align: left
    :width: 210px
    :target: javascript:void(0);

.. include:: ../README.rst
  :start-after:   header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Devices
*******

The Lightning ecosystem provides the following devices:

.. title-card::
    :name: 'lightning.qubit'
    :description: A fast state-vector qubit simulator written in C++
    :link: lightning_qubit/device.html

.. title-card::
    :name: 'lightning.gpu'
    :description: A heterogeneous backend state-vector simulator with NVIDIA cuQuantum library support.
    :link: lightning_gpu/device.html

.. title-card::
    :name: 'lightning.kokkos'
    :description: A heterogeneous backend state-vector simulator with Kokkos library support.
    :link: lightning_kokkos/device.html

.. title-card::
    :name: 'lightning.tensor'
    :description: A heterogeneous backend tensor network simulator with NVIDIA cuQuantum library support.
    :link: lightning_tensor/device.html


Authors
*******

.. include:: ../README.rst
  :start-after: citation-start-inclusion-marker-do-not-remove
  :end-before: citation-end-inclusion-marker-do-not-remove

.. raw:: html

    <div style='clear:both'></div>
    </br>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   dev/installation
   dev/docker
   dev/support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   lightning_qubit/device
   lightning_gpu/device
   lightning_kokkos/device
   lightning_tensor/device

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
