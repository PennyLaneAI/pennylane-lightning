Docker support
**************

Docker images for the various backends are found on the
`PennyLane Docker Hub <https://hub.docker.com/u/pennylaneai>`_ page, where a detailed description about PennyLane Docker support can be found.
Briefly, one can build the Docker Lightning images using:

.. code-block:: bash

    git clone https://github.com/PennyLaneAI/pennylane-lightning.git
    cd pennylane-lightning
    docker build -f docker/Dockerfile --target ${TARGET} .

where ``${TARGET}`` is one of the following

* ``wheel-lightning-qubit``
* ``wheel-lightning-gpu``
* ``wheel-lightning-kokkos-openmp``
* ``wheel-lightning-kokkos-cuda``
* ``wheel-lightning-kokkos-rocm``
* ``wheel-lightning-tensornet-openmp``

To build the Lightning-Tensornet image and run its tests (the Python test suite of the plugin and
the PennyLane device tests, with and without shots), use the ``test-lightning-tensornet-openmp``
target:

.. code-block:: bash

    docker build -f docker/Dockerfile --target test-lightning-tensornet-openmp .

or equivalently ``make docker-test target=lightning-tensornet-openmp``.
