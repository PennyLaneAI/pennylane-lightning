Lightning Tensor device
=======================

The ``lightning.tensor`` device is a tensor network simulator device. The device is built on top of the `cutensornet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`__ from the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum tensor network evolution.

A ``lightning.tensor`` device can be loaded simply using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.tensor", wires=100)

By default, the device represents the quantum state approximated as a Matrix Product State (MPS).
The default setup for the MPS tensor network approximation is:
    - ``max_bond_dim`` (maximum bond dimension) defaults to ``128`` .
    - ``cutoff`` (singular value truncation threshold) defaults to ``0`` .
    - ``cutoff_mode`` (singular value truncation mode) defaults to ``abs`` , considering the absolute values of the singular values; Alternatively, users can opt to set ``cutoff_mode`` to ``rel`` to consider the relative values of the singular values. 

The ``lightning.tensor`` device dispatches all operations to be performed on a CUDA-capable GPU of generation SM 7.0 (Volta)
and greater.

.. note:: 
    Given the inherent parallelism of GPUs, simulations with intensive parallel computation, such as those with larger maximum
    bond dimensions, stand to gain the most from the computational power offered by GPU and those simulations can benifit from the 
    ``lightning.tensor`` device.  It's worth noting that if the bond dimension used in the simulation is small, the ``lightning.tensor`` 
    device with ``MPS`` running a GPU may perform slower compared to a ``default.tensor`` device with ``MPS`` running on a CPU. For more details
    on how bond dimension affects the simulation performance, please refer to the ``Approximate Tensor Network Methods`` section in the `cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`__.

Users also have the flexibility to customize these parameters according to their specific needs with:

.. code-block:: python
    
    import pennylane as qml
    import numpy as np
    
    num_qubits = 100

    device_kwargs_mps = {
        "max_bond_dim": 64,
        "cutoff": 1e-10,
        "cutoff_mode": "abs",
    }

    dev = qml.device("lightning.tensor", wires=num_qubits, method="mps", **device_kwargs_mps)

The ``lightning.tensor`` device allows users to get quantum circuit gradients using the ``parameter-shift`` method. This can be enabled at the PennyLane ``QNode`` level with:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        ...

Check out the :doc:`/lightning_tensor/installation` guide for more information.

.. seealso:: :class:`~DefaultTensor` for a CPU only tensor network simulator device.

Operations and observables support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lightning.tensor`` only supports 1,2-wires gates operations and gates operations that can be decomposed by PennyLane into 1,2-wires gates.  

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
    ~pennylane.ECR
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

**Unsupported operations:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.StatePrep
    ~pennylane.QubitStateVector
    ~pennylane.DoubleExcitationMinus
    ~pennylane.DoubleExcitationPlus
    ~pennylane.GlobalPhase

.. raw:: html

    </div>

**Supported observables:**

The ``lightning.tensor`` supports all observables supported by the Lightning state-vector simulators, besides ``qml.SparseHamiltonian``, ``qml.Projector`` and limited support to ``qml.Hamiltonian``, ``qml.Prod``.

Users can not create a ``Hamiltonian`` or ``Prod`` observable from ``Hamiltonian`` observables.



.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.ops.op_math.Exp
    ~pennylane.Hadamard
    ~pennylane.Hamiltonian
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