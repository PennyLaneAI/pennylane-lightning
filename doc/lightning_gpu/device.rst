Lightning GPU device
====================

The ``lightning.gpu`` device is an extension of PennyLane's built-in ``lightning.qubit`` device.
It extends the CPU-focused Lightning simulator to run using the NVIDIA cuQuantum SDK, enabling GPU-accelerated simulation of quantum state-vector evolution.

A ``lightning.gpu`` device can be loaded using:

.. code-block:: python

    import pennylane as qml
    dev = qml.device("lightning.gpu", wires=2)

If the NVIDIA cuQuantum libraries are available, the above device will allow all operations to be performed on a CUDA capable GPU of generation SM 7.0 (Volta) and greater. If the libraries are not correctly installed, or available on path, the device will fall-back to ``lightning.qubit`` and perform all simulation on the CPU.

The ``lightning.gpu`` device also directly supports quantum circuit gradients using the adjoint differentiation method. This can be enabled at the PennyLane QNode level with:

.. code-block:: python

    @qml.qnode(dev, diff_method="adjoint")
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

    ~pennylane.ops.op_math.Exp
    ~pennylane.Hadamard
    ~pennylane.Hamiltonian
    ~pennylane.Hermitian
    ~pennylane.Identity
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.ops.op_math.Prod
    ~pennylane.Projector
    ~pennylane.SparseHamiltonian
    ~pennylane.ops.op_math.SProd
    ~pennylane.ops.op_math.Sum

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
 
**Multi-GPU/multi-node support:**

The ``lightning.gpu`` device allows users to leverage the computational power of many GPUs sitting on separate nodes for running large-scale simulations. 
Provided that NVIDIA ``cuQuantum`` libraries, a ``CUDA-aware MPI`` library and ``mpi4py`` are properly installed and the path to the ``libmpi.so`` is 
added to the ``LD_LIBRARY_PATH`` environment variable, the following requirements should be met to enable multi-node and multi-GPU simulations:

1. The ``mpi`` keyword argument should be set as ``True`` when initializing a ``lightning.gpu`` device.
2. Both the total number of MPI processes and MPI processes per node must be powers of 2. For example, 2, 4, 8, 16, etc.. Each MPI process is responsible for managing one GPU. 

The workflow for the multi-node/GPUs feature is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane as qml
    dev = qml.device('lightning.gpu', wires=8, mpi=True)
    @qml.qnode(dev)
    def circuit_mpi():
        qml.PauliX(wires=[0])
        return qml.state()
    local_state_vector = circuit_mpi()

Currently, a ``lightning.gpu`` device with the MPI multi-GPU backend supports all the ``gate operations`` and ``observables`` that a ``lightning.gpu`` device with a single GPU/node backend supports.

By default, each MPI process will return the overall simulation results, except for the ``qml.state()`` and ``qml.prob()`` methods for which each MPI process only returns the local simulation
results for the ``qml.state()`` and ``qml.prob()`` methods to avoid buffer overflow. It is the user's responsibility to ensure correct data collection for those two methods. Here are examples of collecting
the local simulation results for ``qml.state()`` and ``qml.prob()`` methods:

The workflow for collecting local state vector (using the ``qml.state()`` method) to ``rank 0`` is as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane as qml
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() 
    dev = qml.device('lightning.gpu', wires=8, mpi=True)
    @qml.qnode(dev)
    def circuit_mpi():
        qml.PauliX(wires=[0])
        return qml.state()
    local_state_vector = circuit_mpi()
    #rank 0 will collect the local state vector
    state_vector = comm.gather(local_state_vector, root=0)
    if rank == 0:
        print(state_vector)
    
The workflow for collecting local probability (using the ``qml.prob()`` method) to ``rank 0`` is as follows:

.. code-block:: python
    
    from mpi4py import MPI
    import pennylane as qml
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    dev = qml.device('lightning.gpu', wires=8, mpi=True)
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

Then the python script can be executed with the following command:

.. code-block:: console
    
    $ mpirun -np 4 python yourscript.py

Furthermore, users can optimize the performance of their applications by allocating the appropriate amount of GPU memory for MPI operations with the ``mpi_buf_size`` keyword argument. To allocate ``n`` mebibytes (MiB, `2^20` bytes) of GPU memory for MPI operations, initialize a ``lightning.gpu`` device with the ``mpi_buf_size=n`` keyword argument, as follows:

.. code-block:: python

    from mpi4py import MPI
    import pennylane as qml
    n = 8
    dev = qml.device("lightning.gpu", wires=20, mpi=True, mpi_buf_size=n)

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
    import pennylane as qml
    from pennylane import numpy as np
  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_wires = 20
    n_layers = 2
  
    dev = qml.device('lightning.gpu', wires= n_wires, mpi=True)
    @qml.qnode(dev, diff_method="adjoint")
    def circuit_adj(weights):
        qml.StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
        return qml.math.hstack([qml.expval(qml.PauliZ(i)) for i in range(n_wires)])
  
    if rank == 0:
        params = np.random.random(qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
    else:
        params = None
  
    params = comm.bcast(params, root=0)
    jac = qml.jacobian(circuit_adj)(params)

If users aim to handle larger system sizes with limited hardware resources, the memory-optimized adjoint method with MPI support is more appropriate. The memory-optimized adjoint method with MPI support employs a single ``bra`` object that is reused for all observables.
This approach results in a notable reduction in the required GPU memory when dealing with a large number of observables. However, it's important to note that the reduction in memory requirement may come at the expense of slower execution due to the multiple ``ket`` updates per gate operation.

To enable the memory-optimized adjoint method with MPI support, ``batch_obs`` should be set as ``True`` and the workflow follows:

.. code-block:: python
    
    dev = qml.device('lightning.gpu', wires= n_wires, mpi=True, batch_obs=True)

For the adjoint method, each MPI process will provide the overall simulation results.
