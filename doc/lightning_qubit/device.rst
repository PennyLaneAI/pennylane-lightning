Lightning Qubit device
======================

The ``lightning.qubit`` device uses a custom-built backend to
perform fast linear algebra calculations for simulating quantum state-vector evolution.

A ``lightning.qubit`` device can be loaded using:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.qubit", wires=2)

Check out the :doc:`/lightning_qubit/installation` guide for more information.

Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane2.BasisState
    ~pennylane2.BlockEncode
    ~pennylane2.CNOT
    ~pennylane2.ControlledPhaseShift
    ~pennylane2.ControlledQubitUnitary
    ~pennylane2.CRot
    ~pennylane2.CRX
    ~pennylane2.CRY
    ~pennylane2.CRZ
    ~pennylane2.CSWAP
    ~pennylane2.CY
    ~pennylane2.CZ
    ~pennylane2.DiagonalQubitUnitary
    ~pennylane2.DoubleExcitation
    ~pennylane2.DoubleExcitationMinus
    ~pennylane2.DoubleExcitationPlus
    ~pennylane2.ECR
    ~pennylane2.GlobalPhase
    ~pennylane2.Hadamard
    ~pennylane2.Identity
    ~pennylane2.IsingXX
    ~pennylane2.IsingXY
    ~pennylane2.IsingYY
    ~pennylane2.IsingZZ
    ~pennylane2.ISWAP
    ~pennylane2.MultiControlledX
    ~pennylane2.MultiRZ
    ~pennylane2.OrbitalRotation
    ~pennylane2.PauliX
    ~pennylane2.PauliY
    ~pennylane2.PauliZ
    ~pennylane2.PauliRot
    ~pennylane2.PCPhase
    ~pennylane2.PhaseShift
    ~pennylane2.PSWAP
    ~pennylane2.QubitCarry
    ~pennylane2.QubitSum
    ~pennylane2.QubitUnitary
    ~pennylane2.Rot
    ~pennylane2.RX
    ~pennylane2.RY
    ~pennylane2.RZ
    ~pennylane2.S
    ~pennylane2.SingleExcitation
    ~pennylane2.SingleExcitationMinus
    ~pennylane2.SingleExcitationPlus
    ~pennylane2.SISWAP
    ~pennylane2.SQISW
    ~pennylane2.SWAP
    ~pennylane2.SX
    ~pennylane2.T
    ~pennylane2.Toffoli

.. raw:: html

    </div>

**Supported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane2.Identity
    ~pennylane2.Hadamard
    ~pennylane2.PauliX
    ~pennylane2.PauliY
    ~pennylane2.PauliZ
    ~pennylane2.Projector
    ~pennylane2.Hermitian
    ~pennylane2.Hamiltonian
    ~pennylane2.SparseHamiltonian
    ~pennylane2.ops.op_math.Exp
    ~pennylane2.ops.op_math.Prod
    ~pennylane2.ops.op_math.SProd
    ~pennylane2.ops.op_math.Sum

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
    import pennylane2 as qp

Assuming you request multiple expectation values from a QNode, this should automatically parallelize the computation over the requested number of threads. You should ensure that the number of threads does not exceed the available physical cores on your machine.

If you are computing a large number of expectation values, or if you are using a large number of wires on your device, it may be best to limit the number of expectation value calculations to at-most ``OMP_NUM_THREADS`` concurrent executions. This will help save memory, at the cost of additional compute time. To enable this, initialize a ``lightning.qubit`` device with the ``batch_obs=True`` keyword argument, as:

.. code-block:: python

    # Before importing packages
    import os
    os.environ["OMP_NUM_THREADS"] = 4
    import pennylane2 as qp
    dev = qp.device("lightning.qubit", wires=2, batch_obs=True)


**OpenMP acceleration of the gate kernels:**

OpenMP acceleration of gate kernels across all kernel types (LM, AVX2, and AVX512) is enabled
by default on Linux and MacOS wheels in Lightning-Qubit.

To learn more about this feature, check out the :doc:`/lightning_qubit/development/kernel_tuning` guide.


**Markov Chain Monte Carlo sampling support:**

The ``lightning.qubit`` device allows users to use the Markov Chain Monte Carlo (MCMC) sampling method to generate approximate samples. To enable the MCMC sampling method for sample generation, initialize a ``lightning.qubit`` device with the ``mcmc=True`` keyword argument, as:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.qubit", wires=2, shots=1000, mcmc=True, kernel_name="Local", num_burnin=100)

By default, ``mcmc=False``, the ``kernel_name`` is ``None`` and ``num_burnin`` is ``0``. The ``Local`` kernel conducts a bit-flip local transition between states. The local kernel generates a random qubit site and then generates a random number to determine  the new bit at that qubit site.

The ``lightning.qubit`` device also supports a ``"NonZeroRandom"`` kernel. This kernel randomly transits between states that have nonzero probability. It can be enabled by initializing the device as:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.qubit", wires=2, shots=1000, mcmc=True, kernel_name="NonZeroRandom", num_burnin=200)
