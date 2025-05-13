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
