Lightning GPU device
====================

The ``lightning.gpu`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to run using the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum state-vector evolution.

A ``lightning.gpu`` device can be loaded using:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.gpu", wires=2)

If the NVIDIA cuQuantum libraries are available, the above device will allow all operations to be performed on a CUDA capable GPU of generation SM 7.0 (Volta) and greater. If the libraries are not correctly installed, or available on path, the device will raise an error.

The ``lightning.gpu`` device supports quantum circuit gradients using the adjoint differentiation method.
By default, this method is enabled. It can also be explicitly specified using the ``diff_method`` argument when creating a device:

.. code-block:: python

    @qp.qnode(dev, diff_method="adjoint")
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

The ``lightning.gpu`` device directly supports the `adjoint differentiation method <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html>`__, and enables parallelization over the requested observables. This supports direct controlling of observable batching, which can be used to run concurrent calculations across multiple available GPUs.

If you are computing a large number of expectation values, or if you are using a large number of wires on your device, it may be best to evenly divide the number of expectation value calculations across all available GPUs. This will reduce the overall memory cost of the observables per GPU, at the cost of additional compute time. Assuming `m` observables, and `n` GPUs, the default behaviour is to pre-allocate all storage for `m` observables on a single GPU. To divide the workload amongst many GPUs, initialize a ``lightning.gpu`` device with the ``batch_obs=True`` keyword argument, as:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.gpu", wires=20, batch_obs=True)

With the above, each GPU will see at most ``ceil(m/n)`` observables to process, reducing the preallocated memory footprint.

Additionally, there can be situations where even with the above distribution, and limited GPU memory, the overall problem does not fit on the requested GPU devices. You can further reduce the concurrent allocations on available GPUs by providing an integer value to the `batch_obs` keyword. For example, to batch evaluate observables with at most 1 observable allocation per GPU, define the device as:

.. code-block:: python

    import pennylane2 as qp
    dev = qp.device("lightning.gpu", wires=27, batch_obs=1)

Each problem is unique, so it can often be best to choose the default behaviour up-front, and tune with the above only if necessary.

**Multi-GPU/multi-node support:**

The ``lightning.gpu`` device allows users to leverage the computational power of many GPUs distributed across multiple nodes for running large-scale simulations.

To utilize distributed simulation, ``lightning.gpu`` must be compiled with MPI support. Check out the :doc:`/lightning_gpu/installation` guide for more information.

With ``lightning.gpu`` installed with MPI support, this can be enabled in Pennylane by setting the ``mpi`` keyword argument to ``True`` when creating the device. For example:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    dev = qp.device('lightning.gpu', wires=8, mpi=True)
    @qp.qnode(dev)
    def circuit_mpi():
        qp.PauliX(wires=[0])
        return qp.state()
    local_state_vector = circuit_mpi()

.. note::

    The total number of MPI processes must be powers of 2, where each MPI process is responsible for managing one GPU.

Currently, a ``lightning.gpu`` device with the MPI multi-GPU backend supports all the ``gate operations`` and ``observables`` that a ``lightning.gpu`` device with a single GPU/node backend supports.

By default, each MPI process will return the overall simulation results, except for the ``qp.state()`` and ``qp.probs()`` methods for which each MPI process only returns the local simulation
results for the ``qp.state()`` and ``qp.probs()`` methods to avoid buffer overflow. It is the user's responsibility to ensure correct data collection for those two methods. Here are examples of collecting
the local simulation results for ``qp.state()`` and ``qp.prob()`` methods:

The workflow for collecting local state vector (using the ``qp.state()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dev = qp.device('lightning.gpu', wires=8, mpi=True)
    @qp.qnode(dev)
    def circuit_mpi():
        qp.PauliX(wires=[0])
        return qp.state()
    local_state_vector = circuit_mpi()
    #rank 0 will collect the local state vector
    state_vector = comm.gather(local_state_vector, root=0)
    if rank == 0:
        print(state_vector)

The workflow for collecting local probability (using the ``qp.prob()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dev = qp.device('lightning.gpu', wires=8, mpi=True)
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

Then the python script can be executed with the following command:

.. code-block:: console

    $ mpirun -np 4 python yourscript.py

Furthermore, users can optimize the performance of their applications by allocating the appropriate amount of GPU memory for MPI operations with the ``mpi_buf_size`` keyword argument. To allocate ``n`` mebibytes (MiB, `2^20` bytes) of GPU memory for MPI operations, initialize a ``lightning.gpu`` device with the ``mpi_buf_size=n`` keyword argument, as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    n = 8
    dev = qp.device("lightning.gpu", wires=20, mpi=True, mpi_buf_size=n)

Note the value of ``mpi_buf_size`` should also be a power of ``2``. Remember to carefully manage the ``mpi_buf_size`` parameter, taking into account the available GPU memory and the memory
requirements of the local state vector, to prevent memory overflow issues and ensure optimal performance. By default (``mpi_buf_size=0``), the GPU memory allocated for MPI operations
will match the size of the local state vector, with a limit of ``64 MiB``. Please be aware that a runtime warning will occur if the local GPU memory buffer for MPI operations exceeds
the GPU memory allocated to the local state vector.

**Multi-GPU/multi-node support for adjoint method:**

The ``lightning.gpu`` device with the multi-GPU/multi-node backend also directly supports the `adjoint differentiation method <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html>`__. Instead of batching observables across the multiple GPUs available within a node, the state vector is distributed among the available GPUs with the multi-GPU/multi-node backend.
By default, the adjoint method with MPI support follows the performance-oriented implementation of the single GPU backend. This means that a separate ``bra`` is created for each observable and the ``ket`` is updated only once for each operation, regardless of the number of observables.

The workflow for the default adjoint method with MPI support is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane2 as qp
    from pennylane2 import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_wires = 20
    n_layers = 2

    dev = qp.device('lightning.gpu', wires= n_wires, mpi=True)
    @qp.qnode(dev, diff_method="adjoint")
    def circuit_adj(weights):
        qp.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
        return qp.math.hstack([qp.expval(qp.PauliZ(i)) for i in range(n_wires)])

    if rank == 0:
        params = np.random.random(qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
    else:
        params = None

    params = comm.bcast(params, root=0)
    jac = qp.jacobian(circuit_adj)(params)

If users aim to handle larger system sizes with limited hardware resources, the memory-optimized adjoint method with MPI support is more appropriate. The memory-optimized adjoint method with MPI support employs a single ``bra`` object that is reused for all observables.
This approach results in a notable reduction in the required GPU memory when dealing with a large number of observables. However, it's important to note that the reduction in memory requirement may come at the expense of slower execution due to the multiple ``ket`` updates per gate operation.

To enable the memory-optimized adjoint method with MPI support, ``batch_obs`` should be set as ``True`` and the workflow follows:

.. code-block:: python

    dev = qp.device('lightning.gpu', wires= n_wires, mpi=True, batch_obs=True)

For the adjoint method, each MPI process will provide the overall simulation results.

.. note::
    The observable ``Projector`` does not have support with the multi-GPU backend.
