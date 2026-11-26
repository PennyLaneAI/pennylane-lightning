Lightning Kokkos device
=======================

The ``lightning.kokkos`` device can run using a variety of HPC-focused backends, including GPUs,
enabling accelerated simulation of quantum state-vector evolution.

A ``lightning.kokkos`` device can be loaded using:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.kokkos", wires=2)

The ``lightning.kokkos`` device also directly supports quantum circuit gradients using the adjoint differentiation method.
By default, this method is enabled. It can also be explicitly specified using the ``diff_method`` argument when creating a device:

.. code-block:: python

    qp.qnode(dev, diff_method="adjoint")
    def circuit(params):
        ...

Check out the :doc:`/lightning_kokkos/installation` guide for more information.

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

**Distributed simulation with MPI:**

The ``lightning.kokkos`` device supports distributed simulation using the Message Passing Interface (MPI). This enables the simulation of larger quantum circuits by distributing the workload across multiple CPU or GPU compute nodes.

To utilize distributed simulation, ``lightning.kokkos`` must be compiled with MPI support. Check out the :ref:`Lightning Kokkos installation guide<install-lightning-kokkos-with-mpi>` for more information.

With ``lightning.kokkos`` installed with MPI support, distributed simulation can be enabled in Pennylane by setting the ``mpi`` keyword argument to ``True`` when creating the device. For example:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp

    dev = qp.device('lightning.kokkos', wires=30, mpi=True)
    @qp.qnode(dev)
    def circuit_mpi():
        qp.PauliX(wires=[0])
        return qp.state()
    local_state_vector = circuit_mpi()

.. note::
    The total number of MPI processes must be powers of 2. If using Kokkos with GPUs, we recommend using one GPU per MPI process.

Currently, the ``lightning.kokkos`` device with MPI supports all the ``gate operations`` and ``observables`` that a single process ``lightning.kokkos`` device supports, excluding ``SparseHamiltonian``.

By default, each MPI process will return the overall simulation results, except for the ``qp.state()`` and ``qp.probs()`` methods for which each MPI process only returns the local simulation
results for the ``qp.state()`` and ``qp.probs()`` methods to avoid buffer overflow. It is the user's responsibility to ensure correct data collection for those two methods (e.g. using MPI Gather). Here are examples of collecting
the local simulation results for ``qp.state()`` and ``qp.probs()`` methods:

The workflow for collecting local state vector (using the ``qp.state()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    dev = qp.device('lightning.kokkos', wires=30, mpi=True)
    @qp.qnode(dev)
    def circuit_mpi():
        qp.PauliX(wires=[0])
        return qp.state()

    local_state_vector = circuit_mpi()

    #rank 0 will collect the local state vector
    state_vector = comm.gather(local_state_vector, root=0)
    if rank == 0:
        print(state_vector)

The workflow for collecting local probability (using the ``qp.probs()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dev = qp.device('lightning.kokkos', wires=30, mpi=True)
    prob_wires = [0, 1]

    @qp.qnode(dev)
    def mpi_circuit():
        qp.Hadamard(wires=1)
        return qp.probs(wires=prob_wires)

    local_probs = mpi_circuit()

    #For data collection across MPI processes.
    recv_counts = comm.gather(len(local_probs),root=0)
    if rank == 0:
        probs = np.zeros(2**len(prob_wires))
    else:
        probs = None

    comm.Gatherv(local_probs,[probs,recv_counts],root=0)
    if rank == 0:
        print(probs)

Then the python script can be executed with the following command (for example on 4 MPI processes):

.. code-block:: console

    $ mpirun -np 4 python pennylane_quantum_script.py
