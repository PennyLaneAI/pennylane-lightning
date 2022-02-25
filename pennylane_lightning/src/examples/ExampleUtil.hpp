#pragma once
#include <cstdlib>
#include <random>
#include <vector>

template <typename RandomEngine>
std::vector<size_t> generateDistinctWires(RandomEngine &re, size_t num_qubits,
                                          size_t num_wires) {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), re);
    return {v.begin(), v.begin() + num_wires};
}

template <typename RandomEngine>
std::vector<size_t> generateNeighboringWires(RandomEngine &re,
                                             size_t num_qubits,
                                             size_t num_wires) {
    std::vector<size_t> v;
    v.reserve(num_wires);
    std::uniform_int_distribution<size_t> idist(0, num_qubits - 1);
    size_t start_idx = idist(re);
    for (size_t k = 0; k < num_wires; k++) {
        v.emplace_back((start_idx + k) % num_qubits);
    }
    return v;
}
