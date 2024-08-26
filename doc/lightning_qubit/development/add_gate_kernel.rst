.. _lightning_add_gate_implementation:

Adding a gate implementation
############################

We discuss how one can add another gate implementation in this document.
Assume that you want to add a custom ``PauliX`` gate implementation in Lightning Qubit.
In this case, you may first create a file and add a class:

.. code-block:: cpp

      // file: MyGateImplementation.hpp
    struct MyGateImplementation {
      public:
        constexpr static std::array implemented_gates = {
            GateOperation::PauliX
        }; // List of implemented gates
        constexpr static kernel_id = KernelType::MyKernel; // Will be discussed below
        constexpr static std::string_view = "MyGateImpl"; // Name of your kernel

        /* This defines the required alignment for this kernel. If there is no special requirement,
           using std::alignment_of_v is sufficient. */
        template <typename PrecisionT>
        constexpr static std::size_t required_alignment = std::alignment_of_v<PrecisionT>;

        template <class PrecisionT>
        static void applyPauliX(std::complex<PrecisionT>* data,
                                std::size_t num_qubits,
                                const std::vector<std::size_t>& wires,
                                [[maybe_unused]] bool inverse) {
            /* Write your implementation */
            ...
        }
    };

Then you can add your gate implementation to Lightning Qubit.
This can be done by modifying two files:

.. code-block:: cpp

    // file: gates/KernelType.hpp
    namespace Pennylane {
    enum class KernelType { PI, LM, MyKernel /* This is added */, None };

    /* Rest of the file */

    } // namespace Pennylane

and

.. code-block:: cpp

    // file: gates/AvailableKernels.hpp
    namespace Pennylane {
        using AvailableKernels = Util::TypeList<GateImplementationsLM,
                                                GateImplementationsPI,
                                                MyGateImplementation /* This is added*/,
                                                void>;
    } // namespace Pennylane


Now you can call your kernel functions in C++.

.. code-block:: cpp

    // sv is a statevector, i.e. an instance of StateVectorRaw or StateVectorManaged

    // call statically
    sv.applyPauliX_<MyKernel>(/*wires=*/{0}, /*inverse=*/false);

    // call using the dynamic dispatcher
    sv.applyOperation(KernelType::MyKernel, "PauliX", /*wires=*/{0}, /*inverse=*/false);

Still, note that your gate implementation is not a default implementation for ``PauliX`` gate yet, i.e.,

.. code-block:: cpp

    // simulator/KernelMap.cpp

    int assignDefaultKernelsForGateOp() {
        auto &instance = OperationKernelMap<GateOperation>::getInstance();

        instance.assignKernelForOp(GateOperation::PauliX, all_threading,
                                   all_memory_model, all_qubit_numbers,
                                   Gates::KernelType::LM);

to

.. code-block:: cpp

    int assignDefaultKernelsForGateOp() {
        auto &instance = OperationKernelMap<GateOperation>::getInstance();

        instance.assignKernelForOp(GateOperation::PauliX, all_threading,
                                   all_memory_model, all_qubit_numbers,
                                   Gates::KernelType::MyKernel);

        ...
    }

will make your implementation as default kernel for ``PauliX`` gate (for all C++ calls as well as for the Python binding).

Gate generators can also be handled in the same way. Note that it is possible to assign the kernel only for specific memory models or
threading operations. Check overloaded functions :cpp:func:`Pennylane::KernelMap::OperationKernelMap::assignKernelForOp` for details.

Test your gate implementation
=============================

To test your own kernel implementations, you can go to ``tests/TestKernels.hpp`` and add your implementation.

.. code-block:: cpp

    using TestKernels = Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                                                  Pennylane::Gates::GateImplementationsPI,
                                                  MyGateImplementation /*This is added */, void>;

It will automatically test your gate implementation.
Note that, in the current implementation, this will test a gate if ``apply + gate name`` is defined even when the gate is not included in ``implemented_gates`` variable.
