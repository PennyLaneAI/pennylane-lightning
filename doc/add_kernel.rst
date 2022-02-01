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

        static void applyPauliY(std::complex<PrecisionT>* data,
                                size_t num_qubits,
                                const std::vector<size_t>& wires,
                                [[maybe_unused]] bool inverse) {
            PL_ABORT("MyGateImplementation::applyPauliY is not implemented");
        }

        /* All other gates */
        ...
    };

Note that all member functions must be defined to prevent compile errors (this requirement may be deprecated in the near future).

Then you can add your gate implementation to Pennylane-Lightning by doing followings:

.. code-block:: cpp

    // file: simulator/KernelType.hpp
    namespace Pennylane {
    enum class KernelType { PI, LM, MyKernel /* This is added */, Unknown };

    namespace Constant {
    constexpr std::array available_kernels = {
        std::pair<KernelType, std::string_view>{KernelType::PI, "PI"},
        std::pair<KernelType, std::string_view>{KernelType::LM, "LM"},
        /* The following line is added */
        std::pair<KernelType, std::string_view>{KernelType::MyKernel, "MyKernel"},
    };

    /* Rest of the file */
    } // namespace Pennylane

and 

.. code-block:: cpp

    // file: simulator/SelectGateOps.hpp
    namespace Pennylane {
        ...
        /* Some code */

        template <class fp_t, KernelType kernel> class SelectGateOps {};

        template <class fp_t>
        class SelectGateOps<fp_t, KernelType::PI> : public GateOperationsPI<fp_t> {};
        template <class fp_t>
        class SelectGateOps<fp_t, KernelType::LM> : public GateOperationsLM<fp_t> {};

        /* Add the following lines */
        template <class fp_t>
        class SelectGateOps<fp_t, KernelType::MyKernel> : public MyGateImplementation<fp_t> {};
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

    // file: simulator/SelectGateOps.hpp
    constexpr std::array<std::pair<GateOperations, KernelType>,
                     static_cast<int>(GateOperations::END)>
    default_kernel_for_ops = {
        std::pair{GateOperations::PauliX, KernelType::LM},
        std::pair{GateOperations::PauliY, KernelType::LM},
        ...
    }

to 

.. code-block:: cpp

    constexpr std::array<std::pair<GateOperations, KernelType>,
                     static_cast<int>(GateOperations::END)>
    default_kernel_for_ops = {
        std::pair{GateOperations::PauliX, KernelType::MyKernel},
        std::pair{GateOperations::PauliY, KernelType::LM},
        ...
    }

will make your implementation as default kernel for ``PauliX`` gate (for all C++ call as well as for the Python binding).
