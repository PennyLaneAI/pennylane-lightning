// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines the class representation for quantum state vectors.
 */

#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Cache.hpp"
#include "Error.hpp"
#include "Gates.hpp"
#include "Util.hpp"

#include <iostream>

/// @cond DEV
namespace {
using namespace std::placeholders;
using std::bind;
using std::size_t;
using std::string;
using std::vector;

// Boost implementation of a hash combine:
// https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
struct hash_function {
    std::size_t
    operator()(const std::pair<std::vector<size_t>, size_t> &key) const {
        std::size_t combined_hash_value = 0;

        for (auto &term : key.first) {
            combined_hash_value ^= std::hash<size_t>()(term) + 0x9e3779b9 +
                                   (combined_hash_value << 6) +
                                   (combined_hash_value >> 2);
        };
        combined_hash_value ^= std::hash<size_t>()(key.second) + 0x9e3779b9 +
                               (combined_hash_value << 6) +
                               (combined_hash_value >> 2);
        return combined_hash_value;
    }
};

}; // namespace
/// @endcond

namespace Pennylane {

/**
 * @brief State-vector operations class.
 *
 * This class binds to a given statevector data array, and defines all
 * operations to manipulate the statevector data for quantum circuit simulation.
 * We define gates as methods to allow direct manipulation of the bound data, as
 * well as through a string-based function dispatch. The bound data is assumed
 * to be complex, and is required to be in either 32-bit (64-bit
 * `complex<float>`) or 64-bit (128-bit `complex<double>`) floating point
 * representation.
 *
 * @tparam fp_t Floating point precision of underlying statevector data.
 */
template <class fp_t = double> class StateVector {
  private:
    using CFP_t = std::complex<fp_t>;

    /***********************************************************************
     *
     * @brief The function dispatching implementation allow the gates to be
     *called directly using a string representation of their names. This enables
     *simplifcation of the call process from Python.
     *
     ***********************************************************************/

    using Func =
        std::function<void(const vector<size_t> &, const vector<size_t> &, bool,
                           const vector<fp_t> &)>;

    using FMap = std::unordered_map<string, Func>;

    //***********************************************************************//

    CFP_t *arr_{nullptr};
    size_t length_{0};
    size_t num_qubits_{0};
    const std::unordered_map<string, size_t> gate_wires_;
    const FMap gates_;

    LRU_cache<std::pair<const std::vector<size_t>, size_t>, std::vector<size_t>,
              hash_function>
        cache_BitPatterns_;

    LRU_cache<std::pair<const std::vector<size_t>, size_t>, std::vector<size_t>,
              hash_function>
        cache_IndicesAfterExclusion_;

  public:
    /**
     * @brief StateVector complex precision type.
     */
    using scalar_type_t = fp_t;

    StateVector()
        : gate_wires_{}, cache_BitPatterns_{}, cache_IndicesAfterExclusion_{} {};

    /**
     * @brief Construct a new `%StateVector` object from a given complex data
     * array.
     *
     * @param arr Pointer to the complex data array.
     * @param length Number of elements in complex data array. Must be
     * power-of-2 (qubits only).
     * @param cache_size Cache size to store most calculated indices throughout
     * operations.
     */
    StateVector(CFP_t *arr, size_t length, size_t cache_size = 10)
        : arr_{arr}, length_{length}, num_qubits_{Util::log2(length_)},
          gate_wires_{
              // Add mapping from function name to required wires.
              {"PauliX", 1},   {"PauliY", 1},     {"PauliZ", 1},
              {"Hadamard", 1}, {"T", 1},          {"S", 1},
              {"RX", 1},       {"RY", 1},         {"RZ", 1},
              {"Rot", 1},      {"PhaseShift", 1}, {"ControlledPhaseShift", 2},
              {"CNOT", 2},     {"SWAP", 2},       {"CZ", 2},
              {"CRX", 2},      {"CRY", 2},        {"CRZ", 2},
              {"CRot", 2},     {"CSWAP", 3},      {"Toffoli", 3}},
          gates_{
              // Add mapping from function name to generalised signature for
              // dispatch. Methods exist with the same signatures to simplify
              // dispatch. Non-parametric gate-calls will ignore the parameter
              // arguments if unused.
              {"PauliX",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyPauliX_(std::forward<decltype(PH1)>(PH1),
                                std::forward<decltype(PH2)>(PH2),
                                std::forward<decltype(PH3)>(PH3),
                                std::forward<decltype(PH4)>(PH4));
               }},
              {"PauliY",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyPauliY_(std::forward<decltype(PH1)>(PH1),
                                std::forward<decltype(PH2)>(PH2),
                                std::forward<decltype(PH3)>(PH3),
                                std::forward<decltype(PH4)>(PH4));
               }},
              {"PauliZ",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyPauliZ_(std::forward<decltype(PH1)>(PH1),
                                std::forward<decltype(PH2)>(PH2),
                                std::forward<decltype(PH3)>(PH3),
                                std::forward<decltype(PH4)>(PH4));
               }},
              {"Hadamard",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyHadamard_(std::forward<decltype(PH1)>(PH1),
                                  std::forward<decltype(PH2)>(PH2),
                                  std::forward<decltype(PH3)>(PH3),
                                  std::forward<decltype(PH4)>(PH4));
               }},
              {"S",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyS_(std::forward<decltype(PH1)>(PH1),
                           std::forward<decltype(PH2)>(PH2),
                           std::forward<decltype(PH3)>(PH3),
                           std::forward<decltype(PH4)>(PH4));
               }},
              {"T",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyT_(std::forward<decltype(PH1)>(PH1),
                           std::forward<decltype(PH2)>(PH2),
                           std::forward<decltype(PH3)>(PH3),
                           std::forward<decltype(PH4)>(PH4));
               }},
              {"CNOT",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCNOT_(std::forward<decltype(PH1)>(PH1),
                              std::forward<decltype(PH2)>(PH2),
                              std::forward<decltype(PH3)>(PH3),
                              std::forward<decltype(PH4)>(PH4));
               }},
              {"SWAP",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applySWAP_(std::forward<decltype(PH1)>(PH1),
                              std::forward<decltype(PH2)>(PH2),
                              std::forward<decltype(PH3)>(PH3),
                              std::forward<decltype(PH4)>(PH4));
               }},
              {"CSWAP",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCSWAP_(std::forward<decltype(PH1)>(PH1),
                               std::forward<decltype(PH2)>(PH2),
                               std::forward<decltype(PH3)>(PH3),
                               std::forward<decltype(PH4)>(PH4));
               }},
              {"CZ",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCZ_(std::forward<decltype(PH1)>(PH1),
                            std::forward<decltype(PH2)>(PH2),
                            std::forward<decltype(PH3)>(PH3),
                            std::forward<decltype(PH4)>(PH4));
               }},
              {"Toffoli",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyToffoli_(std::forward<decltype(PH1)>(PH1),
                                 std::forward<decltype(PH2)>(PH2),
                                 std::forward<decltype(PH3)>(PH3),
                                 std::forward<decltype(PH4)>(PH4));
               }},
              {"PhaseShift",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyPhaseShift_(std::forward<decltype(PH1)>(PH1),
                                    std::forward<decltype(PH2)>(PH2),
                                    std::forward<decltype(PH3)>(PH3),
                                    std::forward<decltype(PH4)>(PH4));
               }},
              {"ControlledPhaseShift",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyControlledPhaseShift_(std::forward<decltype(PH1)>(PH1),
                                              std::forward<decltype(PH2)>(PH2),
                                              std::forward<decltype(PH3)>(PH3),
                                              std::forward<decltype(PH4)>(PH4));
               }},
              {"RX",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyRX_(std::forward<decltype(PH1)>(PH1),
                            std::forward<decltype(PH2)>(PH2),
                            std::forward<decltype(PH3)>(PH3),
                            std::forward<decltype(PH4)>(PH4));
               }},
              {"RY",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyRY_(std::forward<decltype(PH1)>(PH1),
                            std::forward<decltype(PH2)>(PH2),
                            std::forward<decltype(PH3)>(PH3),
                            std::forward<decltype(PH4)>(PH4));
               }},
              {"RZ",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyRZ_(std::forward<decltype(PH1)>(PH1),
                            std::forward<decltype(PH2)>(PH2),
                            std::forward<decltype(PH3)>(PH3),
                            std::forward<decltype(PH4)>(PH4));
               }},
              {"Rot",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyRot_(std::forward<decltype(PH1)>(PH1),
                             std::forward<decltype(PH2)>(PH2),
                             std::forward<decltype(PH3)>(PH3),
                             std::forward<decltype(PH4)>(PH4));
               }},
              {"CRX",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCRX_(std::forward<decltype(PH1)>(PH1),
                             std::forward<decltype(PH2)>(PH2),
                             std::forward<decltype(PH3)>(PH3),
                             std::forward<decltype(PH4)>(PH4));
               }},
              {"CRY",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCRY_(std::forward<decltype(PH1)>(PH1),
                             std::forward<decltype(PH2)>(PH2),
                             std::forward<decltype(PH3)>(PH3),
                             std::forward<decltype(PH4)>(PH4));
               }},
              {"CRZ",
               [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCRZ_(std::forward<decltype(PH1)>(PH1),
                             std::forward<decltype(PH2)>(PH2),
                             std::forward<decltype(PH3)>(PH3),
                             std::forward<decltype(PH4)>(PH4));
               }},
              {"CRot", [this](auto &&PH1, auto &&PH2, auto &&PH3, auto &&PH4) {
                   applyCRot_(std::forward<decltype(PH1)>(PH1),
                              std::forward<decltype(PH2)>(PH2),
                              std::forward<decltype(PH3)>(PH3),
                              std::forward<decltype(PH4)>(PH4));
               }}},cache_BitPatterns_{cache_size}, cache_IndicesAfterExclusion_{
                                              cache_size} {};

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const CFP_t* Pointer to statevector data.
     */
    auto getData() const -> CFP_t * { return arr_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return CFP_t* Pointer to statevector data.
     */
    auto getData() -> CFP_t * { return arr_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data_ptr New data pointer.
     */
    void setData(CFP_t *data_ptr) { arr_ = data_ptr; }

    /**
     * @brief Redefine the length of the statevector and number of qubits.
     *
     * @param length New number of elements in statevector.
     */
    void setLength(size_t length) {
        length_ = length;
        num_qubits_ = Util::log2(length_);
    }
    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t qubits) {
        num_qubits_ = qubits;
        length_ = Util::exp2(num_qubits_);
    }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }

    /**
     * @brief Get the number of qubits represented by the statevector data.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const string &opName, const vector<size_t> &wires,
                        bool inverse = false, const vector<fp_t> &params = {}) {
        const auto gate = gates_.at(opName);
        if (gate_wires_.at(opName) != wires.size()) {
            throw std::invalid_argument(
                string("The gate of type ") + opName + " requires " +
                std::to_string(gate_wires_.at(opName)) + " wires, but " +
                std::to_string(wires.size()) + " were supplied");
        }

        const vector<size_t> internalIndices = generateBitPatterns(wires);
        const vector<size_t> externalWires = getIndicesAfterExclusion(wires);
        const vector<size_t> externalIndices =
            generateBitPatterns(externalWires);
        gate(internalIndices, externalIndices, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param matrix Arbitrary unitary gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::vector<CFP_t> &matrix,
                        const vector<size_t> &wires, bool inverse = false,
                        [[maybe_unused]] const vector<fp_t> &params = {}) {
        auto dim = Util::dimSize(matrix);

        if (dim != wires.size()) {
            throw std::invalid_argument(string("The supplied gate requires ") +
                                        std::to_string(dim) + " wires, but " +
                                        std::to_string(wires.size()) +
                                        " were supplied.");
        }

        const vector<size_t> internalIndices = generateBitPatterns(wires);
        const vector<size_t> externalWires = getIndicesAfterExclusion(wires);
        const vector<size_t> externalIndices =
            generateBitPatterns(externalWires);

        applyMatrix(matrix, internalIndices, externalIndices, inverse);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param wires Vector of wires on which to apply index-matched gate name.
     * @param inverse Indicates whether gate at matched index is to be inverted.
     * @param params Optional parameter data for index matched gates.
     */
    void applyOperations(const vector<string> &ops,
                         const vector<vector<size_t>> &wires,
                         const vector<bool> &inverse,
                         const vector<vector<fp_t>> &params) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size() || numOperations != params.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], wires[i], inverse[i], params[i]);
        }
    }
    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param wires Vector of wires on which to apply index-matched gate name.
     * @param inverse Indicates whether gate at matched index is to be inverted.
     */
    void applyOperations(const vector<string> &ops,
                         const vector<vector<size_t>> &wires,
                         const vector<bool> &inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size()) {
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");
        }

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], wires[i], inverse[i]);
        }
    }

    /**
     * @brief Get indices of statevector data not participating in application
     * operation.
     *
     * @param indicesToExclude Indices to exclude from this call.
     * @param num_qubits Total number of qubits for statevector.
     * @return vector<size_t>
     */
    auto static getIndicesAfterExclusion(const vector<size_t> &indicesToExclude,
                                         size_t num_qubits) -> vector<size_t> {
        std::set<size_t> indices;
        for (size_t i = 0; i < num_qubits; i++) {
            indices.emplace(i);
        }
        for (const size_t &excludedIndex : indicesToExclude) {
            indices.erase(excludedIndex);
        }
        return {indices.begin(), indices.end()};
    }
    /**
     * @brief Get indices of statevector data not participating in application
     operation.
     *
     * @see `getIndicesAfterExclusion(
        const vector<size_t> &indicesToExclude, size_t num_qubits)`
     */
    auto getIndicesAfterExclusion(const vector<size_t> &indicesToExclude)
        -> vector<size_t> {
        return getIndicesAfterExclusion(indicesToExclude, num_qubits_);
    }

    /**
     * @brief Generate indices for applying operations.
     *
     * This method will return the statevector indices participating in the
     * application of a gate to a given set of qubits.
     *
     * @param qubitIndices Indices of the qubits to apply operations.
     * @param num_qubits Number of qubits in register.
     * @return vector<size_t>
     */
    static auto generateBitPatterns(const vector<size_t> &qubitIndices,
                                    size_t num_qubits) -> vector<size_t> {
        vector<size_t> indices;
        indices.reserve(Util::exp2(qubitIndices.size()));
        indices.emplace_back(0);

        for (auto index_it = qubitIndices.rbegin();
             index_it != qubitIndices.rend(); index_it++) {
            const size_t value =
                Util::maxDecimalForQubit(*index_it, num_qubits);
            const size_t currentSize = indices.size();
            for (size_t j = 0; j < currentSize; j++) {
                indices.emplace_back(indices[j] + value);
            }
        }
        return indices;
    }

    /**
     * @brief Generate indices for applying operations.
     *
     * @see `generateBitPatterns(const vector<size_t> &qubitIndices, size_t
     * num_qubits)`.
     */
    auto generateBitPatterns(const vector<size_t> &qubitIndices)
        -> vector<size_t> {
        return generateBitPatterns(qubitIndices, num_qubits_);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Perfect square matrix in row-major order.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(const vector<CFP_t> &matrix, const vector<size_t> &indices,
                     const vector<size_t> &externalIndices, bool inverse) {
        if (static_cast<size_t>(1ULL << (Util::log2(indices.size()) +
                                         Util::log2(externalIndices.size()))) !=
            length_) {
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");
        }

        vector<CFP_t> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            for (size_t i = 0; i < indices.size(); i++) {
                size_t index = indices[i];
                shiftedState[index] = 0;

                if (inverse) {
                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                } else {
                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector read directly
     * from numpy data. Data can be in 1D or 2D format.
     *
     * @param matrix Pointer from numpy data.
     * @param indices Internal indices participating in the operation.
     * @param externalIndices External indices unaffected by the operation.
     * @param inverse Indicate whether inverse should be taken.
     */
    void applyMatrix(const CFP_t *matrix, const vector<size_t> &indices,
                     const vector<size_t> &externalIndices, bool inverse) {
        if (static_cast<size_t>(1ULL << (Util::log2(indices.size()) +
                                         Util::log2(externalIndices.size()))) !=
            length_) {
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");
        }

        vector<CFP_t> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            // Gather
            size_t pos = 0;
            for (const size_t &index : indices) {
                v[pos] = shiftedState[index];
                pos++;
            }

            // Apply + scatter
            for (size_t i = 0; i < indices.size(); i++) {
                size_t index = indices[i];
                shiftedState[index] = 0;

                if (inverse) {
                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            std::conj(matrix[baseIndex + i]) * v[j];
                    }
                } else {
                    const size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * @brief Apply PauliX gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliX(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    /**
     * @brief Apply PauliY gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliY(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            CFP_t v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                -Util::IMAG<fp_t>() * shiftedState[indices[1]];
            shiftedState[indices[1]] = Util::IMAG<fp_t>() * v0;
        }
    }

    /**
     * @brief Apply PauliZ gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyPauliZ(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices,
                     [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= -1;
        }
    }

    /**
     * @brief Apply Hadamard gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyHadamard(const vector<size_t> &indices,
                       const vector<size_t> &externalIndices,
                       [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;

            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<fp_t>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<fp_t>() * (v0 - v1);
        }
    }

    /**
     * @brief Apply S gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyS(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift =
            (inverse) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /**
     * @brief Apply T gate operation to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyT(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift =
            (inverse)
                ? std::conj(std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4))))
                : std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4)));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    /**
     * @brief Apply RX gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRX(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t js = (inverse) ? CFP_t(0, -std::sin(-angle / 2))
                                   : CFP_t(0, std::sin(-angle / 2));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 + js * v1;
            shiftedState[indices[1]] = js * v0 + c * v1;
        }
    }
    /**
     * @brief Apply RY gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRY(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s = (inverse) ? CFP_t(-std::sin(angle / 2), 0)
                                  : CFP_t(std::sin(angle / 2), 0);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }
    /**
     * @brief Apply RZ gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyRZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t first = std::exp(CFP_t(0, -angle / 2));
        const CFP_t second = std::exp(CFP_t(0, angle / 2));
        const CFP_t shift1 = (inverse) ? std::conj(first) : first;
        const CFP_t shift2 = (inverse) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }
    /**
     * @brief Apply phase shift gate operation to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Phase shift angle.
     */
    template <typename Param_t = fp_t>
    void applyPhaseShift(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         Param_t angle) {
        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    /**
     * @brief Apply controlled phase shift gate operation to given indices of
     * statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Phase shift angle.
     */
    template <typename Param_t = fp_t>
    void applyControlledPhaseShift(const std::vector<size_t> &indices,
                                   const std::vector<size_t> &externalIndices,
                                   bool inverse, Param_t angle) {
        const CFP_t s = inverse ? std::conj(std::exp(CFP_t(0, angle)))
                                : std::exp(CFP_t(0, angle));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= s;
        }
    }

    /**
     * @brief Apply Rot gate \f$RZ(\omega)RY(\theta)RZ(\phi)\f$ to given indices
     * of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param phi Gate rotation parameter \f$\phi\f$.
     * @param theta Gate rotation parameter \f$\theta\f$.
     * @param omega Gate rotation parameter \f$\omega\f$.
     */
    template <typename Param_t = fp_t>
    void applyRot(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t phi, Param_t theta, Param_t omega) {
        const vector<CFP_t> rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    /**
     * @brief Apply CNOT (CX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCNOT(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices,
                   [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    /**
     * @brief Apply SWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applySWAP(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices,
                   [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }
    /**
     * @brief Apply CZ gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices,
                 [[maybe_unused]] bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    /**
     * @brief Apply CRX gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRX(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t js = (inverse) ? CFP_t(0, -std::sin(-angle / 2))
                                   : CFP_t(0, std::sin(-angle / 2));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 + js * v1;
            shiftedState[indices[3]] = js * v0 + c * v1;
        }
    }

    /**
     * @brief Apply CRY gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRY(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s = (inverse) ? CFP_t(-std::sin(angle / 2), 0)
                                  : CFP_t(std::sin(angle / 2), 0);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    /**
     * @brief Apply CRZ gate to given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param angle Rotation angle of gate.
     */
    template <typename Param_t = fp_t>
    void applyCRZ(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t m00 = (inverse) ? std::conj(std::exp(CFP_t(0, -angle / 2)))
                                    : std::exp(CFP_t(0, -angle / 2));
        const CFP_t m11 = (inverse) ? std::conj(std::exp(CFP_t(0, angle / 2)))
                                    : std::exp(CFP_t(0, angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    /**
     * @brief Apply CRot gate (controlled \f$RZ(\omega)RY(\theta)RZ(\phi)\f$) to
     * given indices of statevector.
     *
     * @tparam Param_t Precision type for gate parameter. Accepted type are
     * `float` and `double`.
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     * @param phi Gate rotation parameter \f$\phi\f$.
     * @param theta Gate rotation parameter \f$\theta\f$.
     * @param omega Gate rotation parameter \f$\omega\f$.
     */
    template <typename Param_t = fp_t>
    void applyCRot(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse,
                   Param_t phi, Param_t theta, Param_t omega) {
        const auto rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse) ? std::conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse) ? std::conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    /**
     * @brief Apply Toffoli (CCX) gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyToffoli(const vector<size_t> &indices,
                      const vector<size_t> &externalIndices,
                      [[maybe_unused]] bool inverse) {
        // Participating swapped indices
        static const size_t op_idx0 = 6;
        static const size_t op_idx1 = 7;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

    /**
     * @brief Apply CSWAP gate to given indices of statevector.
     *
     * @param indices Local amplitude indices participating in given gate
     * application for fixed sets of non-participating qubit indices.
     * @param externalIndices Non-participating qubit amplitude index offsets
     * for given operation for global application.
     * @param inverse Take adjoint of given operation.
     */
    void applyCSWAP(const vector<size_t> &indices,
                    const vector<size_t> &externalIndices,
                    [[maybe_unused]] bool inverse) {
        // Participating swapped indices
        static const size_t op_idx0 = 5;
        static const size_t op_idx1 = 6;
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[op_idx0]],
                      shiftedState[indices[op_idx1]]);
        }
    }

  private:
    //***********************************************************************//
    //  Internal utility functions for opName dispatch use only.
    //***********************************************************************//
    inline void applyPauliX_(const vector<size_t> &indices,
                             const vector<size_t> &externalIndices,
                             bool inverse, const vector<fp_t> &params) {
        static_cast<void>(params);
        applyPauliX(indices, externalIndices, inverse);
    }
    inline void applyPauliY_(const vector<size_t> &indices,
                             const vector<size_t> &externalIndices,
                             bool inverse, const vector<fp_t> &params) {
        static_cast<void>(params);
        applyPauliY(indices, externalIndices, inverse);
    }
    inline void applyPauliZ_(const vector<size_t> &indices,
                             const vector<size_t> &externalIndices,
                             bool inverse, const vector<fp_t> &params) {
        static_cast<void>(params);
        applyPauliZ(indices, externalIndices, inverse);
    }
    inline void applyHadamard_(const vector<size_t> &indices,
                               const vector<size_t> &externalIndices,
                               bool inverse, const vector<fp_t> &params) {
        static_cast<void>(params);
        applyHadamard(indices, externalIndices, inverse);
    }
    inline void applyS_(const vector<size_t> &indices,
                        const vector<size_t> &externalIndices, bool inverse,
                        const vector<fp_t> &params) {
        static_cast<void>(params);
        applyS(indices, externalIndices, inverse);
    }
    inline void applyT_(const vector<size_t> &indices,
                        const vector<size_t> &externalIndices, bool inverse,
                        const vector<fp_t> &params) {
        static_cast<void>(params);
        applyT(indices, externalIndices, inverse);
    }
    inline void applyRX_(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
        applyRX(indices, externalIndices, inverse, params[0]);
    }
    inline void applyRY_(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
        applyRY(indices, externalIndices, inverse, params[0]);
    }
    inline void applyRZ_(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
        applyRZ(indices, externalIndices, inverse, params[0]);
    }
    inline void applyPhaseShift_(const vector<size_t> &indices,
                                 const vector<size_t> &externalIndices,
                                 bool inverse, const vector<fp_t> &params) {
        applyPhaseShift(indices, externalIndices, inverse, params[0]);
    }
    inline void
    applyControlledPhaseShift_(const vector<size_t> &indices,
                               const vector<size_t> &externalIndices,
                               bool inverse, const vector<fp_t> &params) {
        applyControlledPhaseShift(indices, externalIndices, inverse, params[0]);
    }
    inline void applyRot_(const vector<size_t> &indices,
                          const vector<size_t> &externalIndices, bool inverse,
                          const vector<fp_t> &params) {
        applyRot(indices, externalIndices, inverse, params[0], params[1],
                 params[2]);
    }
    inline void applyCNOT_(const vector<size_t> &indices,
                           const vector<size_t> &externalIndices, bool inverse,
                           const vector<fp_t> &params) {
        static_cast<void>(params);
        applyCNOT(indices, externalIndices, inverse);
    }
    inline void applySWAP_(const vector<size_t> &indices,
                           const vector<size_t> &externalIndices, bool inverse,
                           const vector<fp_t> &params) {
        static_cast<void>(params);
        applySWAP(indices, externalIndices, inverse);
    }
    inline void applyCZ_(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
        static_cast<void>(params);
        applyCZ(indices, externalIndices, inverse);
    }
    inline void applyCRX_(const vector<size_t> &indices,
                          const vector<size_t> &externalIndices, bool inverse,
                          const vector<fp_t> &params) {
        applyCRX(indices, externalIndices, inverse, params[0]);
    }
    inline void applyCRY_(const vector<size_t> &indices,
                          const vector<size_t> &externalIndices, bool inverse,
                          const vector<fp_t> &params) {
        applyCRY(indices, externalIndices, inverse, params[0]);
    }
    inline void applyCRZ_(const vector<size_t> &indices,
                          const vector<size_t> &externalIndices, bool inverse,
                          const vector<fp_t> &params) {
        applyCRZ(indices, externalIndices, inverse, params[0]);
    }
    inline void applyCRot_(const vector<size_t> &indices,
                           const vector<size_t> &externalIndices, bool inverse,
                           const vector<fp_t> &params) {
        applyCRot(indices, externalIndices, inverse, params[0], params[1],
                  params[2]);
    }
    inline void applyToffoli_(const vector<size_t> &indices,
                              const vector<size_t> &externalIndices,
                              bool inverse, const vector<fp_t> &params) {
        static_cast<void>(params);
        applyToffoli(indices, externalIndices, inverse);
    }
    inline void applyCSWAP_(const vector<size_t> &indices,
                            const vector<size_t> &externalIndices, bool inverse,
                            const vector<fp_t> &params) {
        static_cast<void>(params);
        applyCSWAP(indices, externalIndices, inverse);
    }
};
/**
 * @brief Streaming operator for StateVector data.
 *
 * @tparam T StateVector data precision.
 * @param out Output stream.
 * @param sv StateVector to stream.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &out, const StateVector<T> &sv)
    -> std::ostream & {
    const auto length = sv.getLength();
    const auto qubits = sv.getNumQubits();
    const auto data = sv.getData();
    out << "num_qubits=" << qubits << std::endl;
    out << "data=[";
    out << data[0];
    for (size_t i = 1; i < length - 1; i++) {
        out << "," << data[i];
    }
    out << "," << data[length - 1] << "]";

    return out;
}

} // namespace Pennylane
