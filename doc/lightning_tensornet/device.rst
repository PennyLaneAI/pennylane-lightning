Lightning Tensornet device
==========================

The ``lightning.tensornet`` device is a noisy tensor-network simulator built with `Kokkos <https://kokkos.github.io/kokkos-core-wiki/index.html>`_.
It runs on the Kokkos execution space selected at build time (``OPENMP``, ``SERIAL``, ``CUDA``, ``HIP`` or ``SYCL``).

Noiseless circuits are simulated as a pure Matrix Product State (MPS).
The first non-unitary channel switches the simulation, exactly, to a Matrix Product Density Operator (MPDO):
the vectorized density matrix stored as an MPS with physical dimension 4.
Noise can be added to the circuit with PennyLane channels, or supplied to the device as a
:class:`~pennylane.NoiseModel`. The device also supports :func:`~pennylane.qjit` through the
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`_ runtime.

A ``lightning.tensornet`` device can be loaded using:

.. code-block:: python

    import pennylane as qp
    dev = qp.device("lightning.tensornet", wires=100)

The bonds of the MPS are truncated by a singular value decomposition (SVD) after every multi-qubit gate.
The default setup of the truncation is:
    - ``max_bond_dim`` (maximum bond dimension) defaults to ``128``.
    - ``cutoff`` (singular value truncation threshold) defaults to ``0``.
    - ``cutoff_mode`` (singular value truncation mode) defaults to ``"abs"``, which compares the cutoff with the absolute values of the singular values. With ``cutoff_mode="rel"``, the cutoff is relative to the largest singular value of each SVD.

Users can set these options when creating the device:

.. code-block:: python

    import pennylane as qp

    num_qubits = 40

    device_kwargs_mps = {
        "max_bond_dim": 64,
        "cutoff": 1e-10,
        "cutoff_mode": "rel",
    }

    dev = qp.device("lightning.tensornet", wires=num_qubits, **device_kwargs_mps)

The simulation is exact when no singular value is truncated. This is always the case when ``cutoff == 0`` and
``max_bond_dim >= 2**(num_qubits // 2)`` for a pure state, or ``max_bond_dim >= 4**(num_qubits // 2)`` for a noisy state.
After a circuit is executed, the ``truncation_error`` property of the device returns the sum, over all the truncating SVDs,
of the discarded fraction of the squared singular values.

.. note::

    The data type of the simulation can be set with ``c_dtype=np.complex64`` or ``c_dtype=np.complex128`` (default),
    and the Kokkos initialization settings (for example, the number of threads) with the ``kokkos_args`` keyword argument.

Noise models
~~~~~~~~~~~~

A :class:`~pennylane.NoiseModel` is passed to the device with the ``noise_model`` keyword argument:

.. code-block:: python

    import pennylane as qp

    noise = qp.NoiseModel(
        {
            qp.noise.op_in(["CNOT", "CZ"]): qp.noise.partial_wires(qp.DepolarizingChannel, 0.03),
            qp.noise.op_eq("RX"): qp.noise.partial_wires(qp.PhaseDamping, 0.1),
        },
        meas_map={qp.noise.meas_eq(qp.expval): qp.noise.partial_wires(qp.BitFlip, 0.02)},
    )

    dev = qp.device("lightning.tensornet", wires=40, max_bond_dim=64, noise_model=noise)

    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x, 0)
        for i in range(39):
            qp.CNOT([i, i + 1])
        return qp.expval(qp.Z(39))

The ``noise_mode`` keyword argument selects how the noise model is applied:

    - ``"transform"`` (default for a :class:`~pennylane.NoiseModel`): the channels are inserted into each circuit by the device preprocessing, with the same rules as :func:`pennylane.noise.add_noise`, after the circuit has been decomposed to the device gate set.
    - ``"native"``: the noise model is compiled to a native rule table that the C++ backend applies while the circuit runs. The device also accepts a serialized native noise model (a string that starts with ``"LNM1;"``), which can only be used in this mode.

The native mode supports noise models whose channels depend only on the gate name and wires, such as the
models built from :func:`~pennylane.noise.op_eq`, :func:`~pennylane.noise.op_in`, :func:`~pennylane.noise.wires_eq`,
:func:`~pennylane.noise.wires_in` and :func:`~pennylane.noise.partial_wires`. A noise model whose channels depend
on the gate parameters raises an error in the native mode; use ``noise_mode="transform"`` instead.

In both modes, the noise is not applied to state preparation, mid-circuit measurements, or measurements other than
:func:`~pennylane.expval`, :func:`~pennylane.var`, :func:`~pennylane.probs`, :func:`~pennylane.sample` and
:func:`~pennylane.counts`.

Usage with ``qjit``
~~~~~~~~~~~~~~~~~~~

The device can be used in :func:`~pennylane.qjit`-compiled workflows. The compiled program runs on the
Catalyst runtime device ``LightningTensornetSimulator``, which receives the truncation options and the native
form of the noise model of the device:

.. code-block:: python

    import pennylane as qp

    noise = qp.NoiseModel(
        {qp.noise.op_eq(qp.CNOT): qp.noise.partial_wires(qp.DepolarizingChannel, 0.01)}
    )
    dev = qp.device("lightning.tensornet", wires=40, max_bond_dim=64, noise_model=noise)

    @qp.qjit
    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x, 0)
        for i in range(39):
            qp.CNOT([i, i + 1])
        return qp.expval(qp.Z(39))

    circuit(0.3)

.. note::

    Under :func:`~pennylane.qjit`, the noise only comes from the noise model of the device: Catalyst does not
    compile noise channels written in the circuit (for example, :class:`~pennylane.BitFlip`).
    The measurements :func:`~pennylane.density_matrix`, :func:`~pennylane.purity`, :func:`~pennylane.vn_entropy`
    and :func:`~pennylane.mutual_info` are only available without :func:`~pennylane.qjit`.

Measurements and gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lightning.tensornet`` device supports both analytic and finite-shot measurements, as well as mid-circuit
measurements with the ``"device"`` and ``"one-shot"`` methods (``mcm_method`` argument of the :class:`~pennylane.QNode`).

The device is designed for expectation value and variance calculations. Measurements of :func:`~pennylane.probs`
on many wires, and of :func:`~pennylane.state`, return dense vectors of dimension :math:`2^{\text{n_wires}}`,
so they should only be used for small systems (:func:`~pennylane.state` is limited to 30 wires).

:func:`~pennylane.state` is only available while the state is pure. Once a noise channel has been applied,
use :func:`~pennylane.density_matrix` on a subset of the wires instead.

Quantum circuit gradients can be computed with the :func:`~pennylane.gradients.param_shift` and
:func:`~pennylane.gradients.finite_diff` methods, including for noisy circuits:

.. code-block:: python

    @qp.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        ...

Check out the :doc:`/lightning_tensornet/installation` guide for more information.

.. seealso:: :doc:`/lightning_tensor/device` for a GPU tensor-network simulator based on the NVIDIA cuQuantum SDK,
    and `DefaultMixed <https://docs.pennylane.ai/en/latest/code/api/pennylane.devices.default_mixed.DefaultMixed.html>`__
    for a dense density-matrix simulator.

Supported operations and observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    Operations are applied as dense matrices on the MPS. Operations acting on more than 10 wires
    (5 wires once the state is noisy), for example a large :class:`~pennylane.QubitUnitary`, are not supported and raise an error.

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
    ~pennylane.MultiControlledX
    ~pennylane.MultiRZ
    ~pennylane.OrbitalRotation
    ~pennylane.PauliRot
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
    ~pennylane.SISWAP
    ~pennylane.SQISW
    ~pennylane.StatePrep
    ~pennylane.SWAP
    ~pennylane.SX
    ~pennylane.T
    ~pennylane.Toffoli

.. raw:: html

    </div>

**Supported noise channels:**

.. raw:: html

    <div class="summary-table">

.. autosummary::
    :nosignatures:

    ~pennylane.AmplitudeDamping
    ~pennylane.BitFlip
    ~pennylane.DepolarizingChannel
    ~pennylane.GeneralizedAmplitudeDamping
    ~pennylane.PauliError
    ~pennylane.PhaseDamping
    ~pennylane.PhaseFlip
    ~pennylane.QubitChannel
    ~pennylane.ResetError
    ~pennylane.ThermalRelaxationError

.. raw:: html

    </div>

**Supported observables:**

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
    ~pennylane.Projector
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

.. raw:: html

    </div>
