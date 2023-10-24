Lightning-GPU device
======================

The ``lightning.gpu`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to run using the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum state-vector evolution.

A ``lightning.gpu`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=2)

If the NVIDIA cuQuantum libraries are available, the above device will allow all operations to be performed on a CUDA capable GPU of generation SM 7.0 (Volta) and greater. If the libraries are not correctly installed, or available on path, the device will fall-back to ``lightning.qubit`` and perform all simulation on the CPU.

The ``lightning.gpu`` device also directly supports quantum circuit gradients using the adjoint differentiation method. This can be enabled at the PennyLane QNode level with:

.. code-block:: python

    qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        ...

Check out the :doc:`/lightning_gpu/installation` guide for more information.

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
    ~pennylane.CPhase
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
    ~pennylane.QubitStateVector
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

    ~pennylane.Hadamard
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.Hamiltonian
    ~pennylane.SparseHamiltonian
    ~pennylane.Hermitian
    ~pennylane.Sum
    ~pennylane.Prod
    ~pennylane.SProd

.. raw:: html

    </div>



**Parallel adjoint differentiation support:**

The ``lightning.gpu`` device directly supports the `adjoint differentiation method <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html>`__, and enables parallelization over the requested observables. This supports direct controlling of observable batching, which can be used to run concurrent calculations across multiple available GPUs.

If you are computing a large number of expectation values, or if you are using a large number of wires on your device, it may be best to evenly divide the number of expectation value calculations across all available GPUs. This will reduce the overall memory cost of the observables per GPU, at the cost of additional compute time. Assuming `m` observables, and `n` GPUs, the default behaviour is to pre-allocate all storage for `n` observables on a single GPU. To divide the workload amongst many GPUs, initialize a ``lightning.gpu`` device with the ``batch_obs=True`` keyword argument, as:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=20, batch_obs=True)

With the above, each GPU will see at most `m/n` observables to process, reducing the preallocated memory footprint.

Additionally, there can be situations where even with the above distribution, and limited GPU memory, the overall problem does not fit on the requested GPU devices. You can further reduce the concurrent allocations on available GPUs by providing an integer value to the `batch_obs` keyword. For example, to batch evaluate observables with at most 1 observable allocation per GPU, define the device as:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=27, batch_obs=1)

Each problem is unique, so it can often be best to choose the default behaviour up-front, and tune with the above only if necessary.
 