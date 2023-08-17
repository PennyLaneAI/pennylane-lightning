Lightning Kokkos device
=======================

The ``lightning.kokkos`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to run using a variety of HPC focused backends, enabling accelerated simulation of quantum state-vector evolution.

A ``lightning.kokkos`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.kokkos", wires=2)

The ``lightning.kokkos`` device also directly supports quantum circuit gradients using the adjoint differentiation method. This can be enabled at the PennyLane QNode level with:

.. code-block:: python

    qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        ...


Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.CNOT
    ~pennylane.CRot
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.ControlledPhaseShift
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