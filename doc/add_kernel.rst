.. _lightning_add_gate_implementation:

Adding a gate implementation
############################

We discuss how one can add another gate implementation in this document. Assume that you want to add a custom ``PauliX`` gate implementation in Pennylane-Lightning. In this case, you may first add a template class as:

.. code-block:: cpp

    template <class PrecisionT>
    struct MyGateImplementation {
        constexpr static implemented_gates = {GateOperations::PauliX};
        constexpr static kernel_id = KernelType::Mykernel; // Will be discussed below

        static void applyPauliX(std::complex<PrecisionT>* data,
                                size_t num_qubits,
                                const std::vector<size_t>& wires,
                                [[maybe_unused]] bool inverse) {
            /* Write your implementation */
            ...
        }
    };

Then you can add your gate implementation to Pennylane-Lightning. This can be done my modifying two files as:

.. code-block:: cpp

    // file: simulator/KernelType.hpp
    namespace Pennylane {
    enum class KernelType { PI, LM, MyKernel /* This is added */, None };

    /* Rest of the file */

    } // namespace Pennylane

and 

.. code-block:: cpp

    // file: simulator/AvailableKernels.hpp
    namespace Pennylane {
    using AvailableKernels = Util::TypeList<GateImplementationsLM,
                                            GateImplementationsPI,
                                            MyGateImplementation /* This is added*/>;
    } // namespace Pennylane



Now you can call your kernel functions in C++.

.. code-block:: cpp

    // sv is a statevector, i.e. an instance of StateVectorRaw or StateVectorManaged

    // call statically
    sv.applyPauliX_<MyKernel>(/*wires=*/{0}, /*inverse=*/false);

    // call using the dynamic dispatcher
    sv.applyOperation(KernelType::MyKernel, "PauliX", /*wires=*/{0}, /*inverse=*/false);

To export your gate implementation to python, you also need to add your kernel to ``kernels_to_pyexport``:

.. code-block:: cpp

    // file: simulator/KernelType.hpp
    [[maybe_unused]] constexpr std::array kernels_to_pyexport = {
        KernelType::PI, KernelType::LM, KernelType::Mykernel /* This is added */
    };

Then you can find ``PauliX_MyKernel`` function in ``lightning_qubit_ops`` Python module.

Still, note that your gate implementation is not a default implementation for ``PauliX`` gate yet, i.e.,

.. code-block:: cpp

    sv.applyPauliX({0}, false); // still call the default implementation
    sv.applyOperation("PauliX", {0}, false) // still call the default implementation

To make your gate implementation default, you need to change ``default_kernel_for_ops`` constant. Thus changing

.. code-block:: cpp

    // file: simulator/Constant.hpp
    constexpr std::array default_kernel_for_gates = {
        std::pair{GateOperations::PauliX, KernelType::LM},
        std::pair{GateOperations::PauliY, KernelType::LM},
        ...
    }

to 

.. code-block:: cpp

    constexpr std::array default_kernel_for_gates = {
        std::pair{GateOperations::PauliX, KernelType::MyKernel},
        std::pair{GateOperations::PauliY, KernelType::LM},
        ...
    }

will make your implementation as default kernel for ``PauliX`` gate (for all C++ call as well as for the Python binding).
