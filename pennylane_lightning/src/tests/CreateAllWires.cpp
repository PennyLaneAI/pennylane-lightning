#include "CreateAllWires.hpp"

#include <bit>

namespace Pennylane {
auto createAllWires(size_t n_qubits, Gates::GateOperation gate_op, bool order)
    -> std::vector<std::vector<size_t>> {
    if (Util::array_has_elt(Gates::Constant::multi_qubit_gates, gate_op)) {
        // make all possible 2^N permutations
        std::vector<std::vector<size_t>> res;
        res.reserve((1U << n_qubits) - 1);
        for (size_t k = 1; k < (static_cast<size_t>(1U) << n_qubits); k++) {
            std::vector<size_t> wires;
            wires.reserve(std::popcount(k));

            for (size_t i = 0; i < n_qubits; i++) {
                if (((k >> i) & 1U) == 1U) {
                    wires.emplace_back(i);
                }
            }

            res.push_back(wires);
        }
        return res;
    } // else
    const size_t n_wires = Util::lookup(Gates::Constant::gate_wires, gate_op);
    if (order) {
        return PermutationGenerator(n_qubits, n_wires).all_perms();
    } // else
    return CombinationGenerator(n_qubits, n_wires).all_perms();
}
} // namespace Pennylane
