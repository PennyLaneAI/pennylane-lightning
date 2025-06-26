Lightning Kokkos device
=======================

The ``lightning.kokkos`` device can run using a variety of HPC-focused backends, including GPUs,
enabling accelerated simulation of quantum state-vector evolution.

A ``lightning.kokkos`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.kokkos", wires=2)

The ``lightning.kokkos`` device also directly supports quantum circuit gradients using the adjoint differentiation method.
By default, this method is enabled. It can also be explicitly specified using the ``diff_method`` argument when creating a device:

.. code-block:: python

    qml.qnode(dev, diff_method="adjoint")
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

**Distributed simulation with MPI:**

The ``lightning.kokkos`` device supports distributed simulation using the Message Passing Interface (MPI). This enables the simulation of larger quantum circuits by distributing the workload across multiple CPU or GPU compute nodes.

To utilize distributed simulation, ``lightning.kokkos`` must be compiled with MPI support. Check out the :ref:`Lightning Kokkos installation guide<install-lightning-kokkos-with-mpi>` for more information.

With ``lightning.kokkos`` installed with MPI support, distributed simulation can be enabled in Pennylane by setting the ``mpi`` keyword argument to ``True`` when creating the device. For example:

.. code-block:: python

    from mpi4py import MPI
    import pennylane as qml

    dev = qml.device('lightning.kokkos', wires=8, mpi=True)
    @qml.qnode(dev)
    def circuit_mpi():
        qml.PauliX(wires=[0])
        return qml.state()
    local_state_vector = circuit_mpi()

.. note::
    The total number of MPI processes must be powers of 2. For example, 2, 4, 8, 16, etc. If using Kokkos with GPUs, we recommend using one GPU per MPI process.

Currently, the ``lightning.kokkos`` device with MPI supports all the ``gate operations`` and ``observables`` that a single process ``lightning.kokkos`` device supports, excluding ``SparseHamiltonian``.

By default, each MPI process will return the overall simulation results, except for the ``qml.state()`` and ``qml.probs()`` methods for which each MPI process only returns the local simulation
results for the ``qml.state()`` and ``qml.probs()`` methods to avoid buffer overflow. It is the user's responsibility to ensure correct data collection for those two methods (e.g. using MPI Gather). Here are examples of collecting
the local simulation results for ``qml.state()`` and ``qml.probs()`` methods:

The workflow for collecting local state vector (using the ``qml.state()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane as qml
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 

    dev = qml.device('lightning.kokkos', wires=8, mpi=True)
    @qml.qnode(dev)
    def circuit_mpi():
        qml.PauliX(wires=[0])
        return qml.state()

    local_state_vector = circuit_mpi()

    #rank 0 will collect the local state vector
    state_vector = comm.gather(local_state_vector, root=0)
    if rank == 0:
        print(state_vector)
    
The workflow for collecting local probability (using the ``qml.probs()`` method) to ``rank 0`` is as follows:

.. code-block:: python
    
    from mpi4py import MPI
    import pennylane as qml
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dev = qml.device('lightning.kokkos', wires=8, mpi=True)
    prob_wires = [0, 1]

    @qml.qnode(dev)
    def mpi_circuit():
        qml.Hadamard(wires=1)
        return qml.probs(wires=prob_wires)

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
