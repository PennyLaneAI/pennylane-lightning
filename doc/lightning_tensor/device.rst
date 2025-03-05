Lightning Tensor device
=======================

The ``lightning.tensor`` device is a tensor network simulator, supporting both the Matrix Product State (MPS) and Exact Tensor Network (TN) methods. The device is built on top of the `cutensornet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`__ library from the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum tensor network evolution. This device is designed to simulate large-scale quantum circuits using tensor networks. For small circuits, state-vector simulator plugins may be more suitable.

The ``lightning.tensor`` device defaults to the Matrix Product State (MPS) method, and can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.tensor", wires=100)

By default, the device approximates the quantum state using an MPS.
The default setup for the MPS tensor network approximation is:
    - ``max_bond_dim`` (maximum bond dimension) defaults to ``128`` .
    - ``cutoff`` (singular value truncation threshold) defaults to ``0`` .
    - ``cutoff_mode`` (singular value truncation mode) defaults to ``abs`` , considering the absolute values of the singular values; Alternatively, users can opt to set ``cutoff_mode`` to ``rel`` to consider the relative values of the singular values.
Note that ``cutensornet`` will automatically determine the reduced extent of the bond dimension based on the lowest among the multiple truncation cutoffs (``max_bond_dim``, ``cutoff-abs`` or ``cutoff-rel``). For more details on how the ``cutoff`` works, please check the `cuQuantum documentation <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornettensorsvdconfigattributes-t>`__.

Users also have the flexibility to customize MPS parameters according to their specific needs with:

.. code-block:: python
    
    import pennylane as qml
    
    num_qubits = 100

    device_kwargs_mps = {
        "max_bond_dim": 64,
        "cutoff": 1e-10,
        "cutoff_mode": "abs",
    }

    dev = qml.device("lightning.tensor", wires=num_qubits, method="mps", **device_kwargs_mps)

Users can also run the ``lightning.tensor`` device in the **Exact Tensor Network** mode by setting the ``method`` argument to ``"tn"``:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.tensor", wires=100, method="tn")

The ``lightning.tensor`` device dispatches all operations to be performed on a CUDA-capable GPU of generation SM 7.0 
and greater (Volta and later). This device supports both exact and finite shots measurements. Currently, the supported differentiation methods are parameter-shift and finite-diff. Note that the MPS backend of ``lightning.tensor`` supports multi-wire gates via Matrix Product Operators (MPO).

The ``lightning.tensor`` device is designed for expectation value calculations. Measurements of :func:`~pennylane.probs` or :func:`~pennylane.state` return dense vectors of dimension :math:`2^{\text{n_qubits}}`, so they should only be used for small systems.

.. note:: Currently only single-wire :class:`~pennylane.Hermitian` observables are supported. You can use :func:`~pennylane.pauli_decompose` on smaller matrices to obtain a compatible Pauli decomposition in the meantime.

The ``lightning.tensor`` device allows users to get quantum circuit gradients using the :func:`~pennylane.gradients.param_shift` method. This can be enabled at the PennyLane :class:`~pennylane.QNode` level with:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        ...

Check out the :doc:`/lightning_tensor/installation` guide for more information.

Note that ``lightning.tensor`` cannot be cleaned up like other state-vector devices since the data is attached to the graph. It is recommended to create a new ``lightning.tensor`` device per circuit to ensure resources are correctly handled.

.. seealso:: `DefaultTensor <https://docs.pennylane.ai/en/latest/code/api/pennylane.devices.default_tensor.DefaultTensor.html>`__ for a CPU only tensor network simulator device.


Operations and observables support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lightning.tensor`` supports all gate operations supported by PennyLane, with the exception of :class:`~pennylane.StatePrep`, which is **not supported** by the **Exact Tensor Network** method. 

**Supported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.BlockEncode
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
    ~pennylane.GlobalPhase
    ~pennylane.Hadamard
    ~pennylane.Identity
    ~pennylane.IsingXX
    ~pennylane.IsingXY
    ~pennylane.IsingYY
    ~pennylane.IsingZZ
    ~pennylane.ISWAP
    ~pennylane.OrbitalRotation
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.PSWAP
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
    ~pennylane.StatePrep
    ~pennylane.SISWAP
    ~pennylane.SQISW
    ~pennylane.SWAP
    ~pennylane.SX
    ~pennylane.T
    ~pennylane.Toffoli

.. raw:: html

    </div>


**Supported observables:**

The ``lightning.tensor`` supports all observables supported by the Lightning state-vector simulators, besides :class:`~pennylane.SparseHamiltonian`, :class:`~pennylane.Projector` and limited support to :class:`~pennylane.ops.op_math.Sum`, :class:`~pennylane.ops.op_math.Prod` since ``lightning.tensor`` only supports 1-wire :class:`~pennylane.Hermitian` observables.

Users cannot create a :class:`~pennylane.ops.op_math.Sum` observable or a :class:`~pennylane.ops.op_math.Prod` observable from :class:`~pennylane.ops.op_math.Sum` observables.



.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.ops.op_math.Exp
    ~pennylane.Hadamard
    ~pennylane.Hermitian
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.ops.op_math.Prod
    ~pennylane.ops.op_math.SProd
    ~pennylane.ops.op_math.Sum

.. raw:: html

    </div>

**Unsupported observables:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.SparseHamiltonian
    ~pennylane.Projector

.. raw:: html

    </div>
