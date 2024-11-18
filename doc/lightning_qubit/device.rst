Lightning Qubit device
======================

The ``lightning.qubit`` device uses a custom-built backend to
perform fast linear algebra calculations for simulating quantum state-vector evolution.

A ``lightning.qubit`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=2)

Check out the :doc:`/lightning_qubit/installation` guide for more information.

Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.CNOT
    ~pennylane.ControlledPhaseShift
    ~pennylane.ControlledQubitUnitary
    ~pennylane.CRot
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.CSWAP
    ~pennylane.CY
    ~pennylane.CZ
    ~pennylane.DiagonalQubitUnitary
    ~pennylane.DoubleExcitation
    ~pennylane.DoubleExcitationMinus
    ~pennylane.DoubleExcitationPlus
    ~pennylane.ECR
    ~pennylane.Hadamard
    ~pennylane.Identity
    ~pennylane.IsingXX
    ~pennylane.IsingXY
    ~pennylane.IsingYY
    ~pennylane.IsingZZ
    ~pennylane.ISWAP
    ~pennylane.MultiControlledX
    ~pennylane.MultiRZ
    ~pennylane.OrbitalRotation
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.PSWAP
    ~pennylane.QFT
    ~pennylane.QubitCarry
    ~pennylane.QubitSum
    ~pennylane.QubitUnitary
    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.S
    ~pennylane.SingleExcitation
    ~pennylane.SingleExcitationMinus
    ~pennylane.SingleExcitationPlus
    ~pennylane.SISWAP
    ~pennylane.SQISW
    ~pennylane.SWAP
    ~pennylane.SX
    ~pennylane.T
    ~pennylane.Toffoli

.. raw:: html

    </div>

**Supported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.Identity
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.Projector
    ~pennylane.Hermitian
    ~pennylane.Hamiltonian
    ~pennylane.SparseHamiltonian
    ~pennylane.ops.op_math.Exp
    ~pennylane.ops.op_math.Prod
    ~pennylane.ops.op_math.SProd
    ~pennylane.ops.op_math.Sum

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


**Markov Chain Monte Carlo sampling support:**

The ``lightning.qubit`` device allows users to use the Markov Chain Monte Carlo (MCMC) sampling method to generate approximate samples. To enable the MCMC sampling method for sample generation, initialize a ``lightning.qubit`` device with the ``mcmc=True`` keyword argument, as:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=2, shots=1000, mcmc=True)

By default, the ``kernel_name`` is ``"Local"`` and ``num_burnin`` is ``100``. The local kernel conducts a bit-flip local transition between states. The local kernel generates a random qubit site and then generates a random number to determine  the new bit at that qubit site.

The ``lightning.qubit`` device also supports a ``"NonZeroRandom"`` kernel. This kernel randomly transits between states that have nonzero probability. It can be enabled by initializing the device as:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=2, shots=1000, mcmc=True, kernel_name="NonZeroRandom", num_burnin=200)

