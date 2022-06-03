#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "GateOperation.hpp"
#include "Util.hpp"

namespace Pennylane::Gates {
template <typename T, size_t size1, size_t size2>
constexpr auto are_mutually_disjoint(const std::array<T, size1> &arr1,
                                     const std::array<T, size2> &arr2) -> bool {
    return std::all_of(arr1.begin(), arr1.end(), [&arr2](const auto &elt) {
        return !Util::array_has_elt(arr2, elt);
    });
}

/*******************************************************************************
 * Check gate_names is well defined
 ******************************************************************************/

static_assert(Constant::gate_names.size() ==
                  static_cast<size_t>(GateOperation::END),
              "Constant gate_names must be defined for all gate operations.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::gate_names)) ==
                  Constant::gate_names.size(),
              "First elements of gate_names must be distinct.");
static_assert(Util::count_unique(Util::second_elts_of(Constant::gate_names)) ==
                  Constant::gate_names.size(),
              "Second elements of gate_names must be distinct.");

/*******************************************************************************
 * Check generator_names is well defined
 ******************************************************************************/

constexpr auto check_generator_names_starts_with() -> bool {
    const auto &arr = Constant::generator_names;
    return std::all_of(arr.begin(), arr.end(), [](const auto &elt) {
        const auto &[gntr_op, gntr_name] = elt;
        return gntr_name.substr(0, 9) == "Generator";
    });
    return true;
}

static_assert(
    Constant::generator_names.size() ==
        static_cast<size_t>(GeneratorOperation::END),
    "Constant generator_names must be defined for all generator operations.");
static_assert(
    Util::count_unique(Util::first_elts_of(Constant::generator_names)) ==
        Constant::generator_names.size(),
    "First elements of generator_names must be distinct.");
static_assert(
    Util::count_unique(Util::second_elts_of(Constant::generator_names)) ==
        Constant::generator_names.size(),
    "Second elements of generator_names must be distinct.");
static_assert(check_generator_names_starts_with(),
              "Names of generators must start with \"Generator\"");

/*******************************************************************************
 * Check gate_wires is well defined
 ******************************************************************************/

static_assert(Constant::gate_wires.size() ==
                  static_cast<size_t>(GateOperation::END) -
                      Constant::multi_qubit_gates.size(),
              "Constant gate_wires must be defined for all gate operations "
              "acting on a fixed number of qubits.");
static_assert(
    are_mutually_disjoint(Util::first_elts_of(Constant::gate_wires),
                          Constant::multi_qubit_gates),
    "Constant gate_wires must not define values for multi-qubit gates.");
static_assert(Util::count_unique(Util::first_elts_of(Constant::gate_wires)) ==
                  Constant::gate_wires.size(),
              "First elements of gate_wires must be distinct.");

/*******************************************************************************
 * Check generator_wires is well defined
 ******************************************************************************/

static_assert(
    Constant::generator_wires.size() ==
        static_cast<size_t>(GeneratorOperation::END) -
            Constant::multi_qubit_generators.size(),
    "Constant generator_wires must be defined for all generator operations "
    "acting on a fixed number of qubits.");
static_assert(
    are_mutually_disjoint(Util::first_elts_of(Constant::generator_wires),
                          Constant::multi_qubit_generators),
    "Constant generator_wires must not define values for multi-qubit "
    "generators.");
static_assert(
    Util::count_unique(Util::first_elts_of(Constant::generator_wires)) ==
        Constant::generator_wires.size(),
    "First elements of generator_wires must be distinct.");
} // namespace Pennylane::Gates
