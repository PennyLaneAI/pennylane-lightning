Lightning Tensor device
=======================

The ``lightning.tensor`` device is a tensor network based device. It extends Pennylane-Lightning to run tensor network based simulations. 
The device is built on top of the `cutensornet` from the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum tensor network evolution.

A ``lightning.tensor`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.tensor", wires=100)

If the NVIDIA cuQuantum libraries are available, the above device will allow all operations to be performed on a CUDA capable GPU of generation SM 7.0 (Volta) and greater. 

The ``lightning.tensor`` device also directly supports quantum circuit gradients using the ``parameter-shift`` method. This can be enabled at the PennyLane QNode level with:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        ...

Check out the :doc:`/lightning_tensor/installation` guide for more information.

Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lightning.tensor`` only supports 1,2-wires gates operations and gates operations that can be decomposed into 1,2-wires gates.

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

**Supported observables:**

The ``lightning.tensor`` supports all observables supported by lightning state-vector simulators, besides ``qml.SparseHamiltonian`` and ``qml.Projector``.


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
