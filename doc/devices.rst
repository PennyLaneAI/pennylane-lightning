Lightning Qubit
===============

The ``lightning.qubit`` device is an extension of PennyLane's in-built ``default.qubit`` device.
It uses the C++ Eigen library to perform fast linear algebra calculations such as the tensor
contraction, which is the main calculation for a state-vector simulator.

A ``lightning.qubit`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lighting.qubit", wires=2)
