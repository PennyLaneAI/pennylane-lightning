#include "GateOperation.hpp"
#include "Constant.hpp"
#include "Util.hpp"

namespace Pennylane {

template <typename T, size_t size1, size_t size2>
constexpr auto are_mutually_disjoint(const std::array<T, size1>& arr1,
                                const std::array<T, size2>& arr2) -> bool {
    for(const T& elt: arr2) {
        if (Util::array_has_elt(arr1, elt)) {
            return false;
        }
    }
    return true;
}

/*******************************************************************************
 * Check gate_names is well defined
 ******************************************************************************/

static_assert(Constant::gate_names.size() == 
        static_cast<size_t>(GateOperation::END), 
        "Cosntant gate_names must be defined for all gate operations.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::gate_names)) == 
        static_cast<size_t>(GateOperation::END), 
        "First elements of gate_names must be distinct.");
static_assert(Util::count_unique(Util::second_elts_of(Constant::gate_names)) == 
        static_cast<size_t>(GateOperation::END), 
        "Secondelements of gate_names must be distinct.");

/*******************************************************************************
 * Check generator_names is well defined
 ******************************************************************************/

static_assert(Constant::generator_names.size() == 
        static_cast<size_t>(GeneratorOperation::END), 
        "Cosntant generator_names must be defined for all generator operations.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::generator_names)) == 
        static_cast<size_t>(GeneratorOperation::END), 
        "First elements of generator_names must be distinct.");
static_assert(Util::count_unique(Util::second_elts_of(Constant::generator_names)) == 
        static_cast<size_t>(GeneratorOperation::END), 
        "Secondelements of generator_names must be distinct.");

/*******************************************************************************
 * Check gate_wires is well defined
 ******************************************************************************/

static_assert(Constant::gate_wires.size() == 
        static_cast<size_t>(GateOperation::END) - Constant::multi_qubit_gates.size(), 
        "Cosntant gate_wires must be defined for all gate operations "
        "acting on a fixed number of qubits.");
static_assert(are_mutually_disjoint(Util::first_elts_of(Constant::gate_wires),
            Constant::multi_qubit_gates), 
        "Constant gate_wires must not define values for multi-qubit gates.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::gate_wires)) == 
        Constant::gate_wires.size(), 
        "First elements of gate_wires must be distinct.");

/*******************************************************************************
 * Check generator_wires is well defined
 ******************************************************************************/

static_assert(Constant::generator_wires.size() == 
        static_cast<size_t>(GeneratorOperation::END) - Constant::multi_qubit_generators.size(), 
        "Cosntant generator_wires must be defined for all generator operations "
        "acting on a fixed number of qubits.");
static_assert(are_mutually_disjoint(Util::first_elts_of(Constant::generator_wires),
            Constant::multi_qubit_generators), 
        "Constant generator_wires must not define values for multi-qubit generators.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::generator_wires)) == 
        Constant::generator_wires.size(), 
        "First elements of generator_wires must be distinct.");

/*******************************************************************************
 * Check default_kernel_for_gates are defined for all gates
 ******************************************************************************/

static_assert(Util::count_unique(Util::first_elts_of(
                Constant::default_kernel_for_gates)) ==
        static_cast<size_t>(GateOperation::END));

/*******************************************************************************
 * Check default_kernel_for_generators are defined for all generators
 ******************************************************************************/

static_assert(Util::count_unique(Util::first_elts_of(
                Constant::default_kernel_for_generators)) ==
        static_cast<size_t>(GeneratorOperation::END));

} // namespace Pennylane
