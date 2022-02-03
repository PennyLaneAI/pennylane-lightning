Lightning Qubit device
======================

The ``lightning.qubit`` device is an extension of PennyLane's built-in ``default.qubit`` device.
It uses a custom-built backend to
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


**Parallel adjoint differentiation support:**

The ``lightning.qubit`` device directly supports the `adjoint differentiation method <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html>`__, and enables parallelization over the requested observables (Linux/MacOS support only).

To enable parallel differentiation over observables, ensure the ``OMP_NUM_THREADS`` environment variable is set before starting your Python session, or if already started, before importing packages:

.. code-block:: bash

    # Option 1: Before starting Python
    export OMP_NUM_THREADS=4
    python <your_file>.py

.. code-block:: python

    # Option 2: Before importing packages
    import os
    os.environ["OMP_NUM_THREADS"] = 4
    import pennylane as qml

Assuming you request multiple expectation values from a QNode, this should automatically parallelize the computation over the requested number of threads. You should ensure that the number of threads does not exceed the available physical cores on your machine.

If you are computing a large number of expectation values, or if you are using a large number of wires on your device, it may be best to limit the number of expectation value calculations to at-most ``OMP_NUM_THREADS`` concurrent executions. This will help save memory, at the cost of additional compute time. To enable this, initialize a ``lightning.qubit`` device with the ``batch_obs=True`` keyword argument, as:

.. code-block:: python

    # Before importing packages
    import os
    os.environ["OMP_NUM_THREADS"] = 4
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=2, batch_obs=True)
