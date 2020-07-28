Lightning Qubit device
======================

The ``lightning.qubit`` device is an extension of PennyLane's built-in ``default.qubit`` device.
It uses the C++ Eigen library to perform fast linear algebra calculations for simulating quantum state-vector evolution.

A ``lightning.qubit`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lighting.qubit", wires=2)
