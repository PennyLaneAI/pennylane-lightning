Lightning Qubit device
======================

The ``lightning.qubit`` device is an extension of PennyLane's built-in ``default.qubit`` device.
It uses a custom-build backend to
perform fast linear algebra calculations for simulating quantum state-vector evolution.

A ``lightning.qubit`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=2)

Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.CNOT
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.QubitStateVector
    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.S
    ~pennylane.T

.. raw:: html

    </div>

**Supported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Hadamard
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ

.. raw:: html

    </div>
