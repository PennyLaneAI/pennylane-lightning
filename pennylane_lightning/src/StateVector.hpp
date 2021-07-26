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

// Required for compilation with MSVC
#define _USE_MATH_DEFINES

#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Gates.hpp"
#include "Util.hpp"

namespace {
using namespace std::placeholders;
using std::bind;
using std::size_t;
using std::string;
using std::vector;
}; // namespace

namespace Pennylane {

template <class fp_t = double> class StateVector {
  private:
    using CFP_t = std::complex<fp_t>;

    using Func =
        std::function<void(const vector<size_t> &, const vector<size_t> &, bool,
                           const vector<fp_t> &)>;

    using FMap = std::unordered_map<string, Func>;

    const FMap gates_;
    const std::unordered_map<string, size_t> gate_wires_;

    CFP_t *const arr_;
    const size_t length_;
    const size_t num_qubits_;

  public:
    StateVector()
        : arr_{nullptr}, length_{0}, num_qubits_{0}, gate_wires_{}, gates_{} {};
    StateVector(CFP_t *arr, size_t length)
        : arr_{arr}, length_{length}, num_qubits_{Util::log2(length_)},
          gate_wires_{
              {"PauliX", 1}, {"PauliY", 1}, {"PauliZ", 1},     {"Hadamard", 1},
              {"T", 1},      {"S", 1},      {"RX", 1},         {"RY", 1},
              {"RZ", 1},     {"Rot", 1},    {"PhaseShift", 1}, {"CNOT", 2},
              {"SWAP", 2},   {"CZ", 2},     {"CRX", 2},        {"CRY", 2},
              {"CRZ", 2},    {"CRot", 2},   {"CSWAP", 3},      {"Toffoli", 3}},
          gates_{
              {"PauliX",
               bind(&StateVector<fp_t>::applyPauliX_, this, _1, _2, _3, _4)},
              {"PauliY",
               bind(&StateVector<fp_t>::applyPauliY_, this, _1, _2, _3, _4)},
              {"PauliZ",
               bind(&StateVector<fp_t>::applyPauliZ_, this, _1, _2, _3, _4)},
              {"Hadamard",
               bind(&StateVector<fp_t>::applyHadamard_, this, _1, _2, _3, _4)},
              {"S", bind(&StateVector<fp_t>::applyS_, this, _1, _2, _3, _4)},
              {"T", bind(&StateVector<fp_t>::applyT_, this, _1, _2, _3, _4)},
              {"CNOT",
               bind(&StateVector<fp_t>::applyCNOT_, this, _1, _2, _3, _4)},
              {"SWAP",
               bind(&StateVector<fp_t>::applySWAP_, this, _1, _2, _3, _4)},
              {"CSWAP",
               bind(&StateVector<fp_t>::applyCSWAP_, this, _1, _2, _3, _4)},
              {"CZ", bind(&StateVector<fp_t>::applyCZ_, this, _1, _2, _3, _4)},
              {"Toffoli",
               bind(&StateVector<fp_t>::applyToffoli_, this, _1, _2, _3, _4)},
              {"PhaseShift", bind(&StateVector<fp_t>::applyPhaseShift_, this,
                                  _1, _2, _3, _4)},
              {"RX", bind(&StateVector<fp_t>::applyRX_, this, _1, _2, _3, _4)},
              {"RY", bind(&StateVector<fp_t>::applyRY_, this, _1, _2, _3, _4)},
              {"RZ", bind(&StateVector<fp_t>::applyRZ_, this, _1, _2, _3, _4)},
              {"Rot",
               bind(&StateVector<fp_t>::applyRot_, this, _1, _2, _3, _4)},
              {"CRX",
               bind(&StateVector<fp_t>::applyCRX_, this, _1, _2, _3, _4)},
              {"CRY",
               bind(&StateVector<fp_t>::applyCRY_, this, _1, _2, _3, _4)},
              {"CRZ",
               bind(&StateVector<fp_t>::applyCRZ_, this, _1, _2, _3, _4)},
              {"CRot",
               bind(&StateVector<fp_t>::applyCRot_, this, _1, _2, _3, _4)}} {};

    CFP_t *getData() { return arr_; }
    std::size_t getLength() { return length_; }

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
        if (gate_wires_.at(opName) != wires.size())
            throw std::invalid_argument(
                string("The gate of type ") + opName + " requires " +
                std::to_string(gate_wires_.at(opName)) + " wires, but " +
                std::to_string(wires.size()) + " were supplied");

        const vector<size_t> internalIndices = generateBitPatterns(wires);
        const vector<size_t> externalWires = getIndicesAfterExclusion(wires);
        const vector<size_t> externalIndices =
            generateBitPatterns(externalWires);

        gate(internalIndices, externalIndices, inverse, params);
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
        if (numOperations != wires.size() || numOperations != params.size())
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");

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
     * @param params Optional parameter data for index matched gates.
     */
    void applyOperations(const vector<string> &ops,
                         const vector<vector<size_t>> &wires,
                         const vector<bool> &inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != wires.size())
            throw std::invalid_argument(
                "Invalid arguments: number of operations, wires, and "
                "parameters must all be equal");

        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], wires[i], inverse[i]);
        }
    }

    /**
     * @brief Get indices not participating in operation.
     *
     * @param indicesToExclude
     * @return vector<size_t>
     */
    vector<size_t> static getIndicesAfterExclusion(
        const vector<size_t> &indicesToExclude, size_t num_qubits) {
        std::set<size_t> indices;
        for (size_t i = 0; i < num_qubits; i++) {
            indices.emplace(i);
        }
        for (const size_t &excludedIndex : indicesToExclude) {
            indices.erase(excludedIndex);
        }
        return {indices.begin(), indices.end()};
    }
    vector<size_t>
    getIndicesAfterExclusion(const vector<size_t> &indicesToExclude) {
        return getIndicesAfterExclusion(indicesToExclude, num_qubits_);
    }

    /**
     * @brief Generate bit patterns for applying operations.
     *
     * @param qubitIndices Indices of the qubits to apply operations.
     * @return vector<size_t>
     */
    static vector<size_t>
    generateBitPatterns(const vector<size_t> &qubitIndices, size_t num_qubits) {
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

    vector<size_t> generateBitPatterns(const vector<size_t> &qubitIndices) {
        return generateBitPatterns(qubitIndices, num_qubits_);
    }

    // Apply Gates
    void applyUnitary(const vector<CFP_t> &matrix,
                      const vector<size_t> &indices,
                      const vector<size_t> &externalIndices, bool inverse) {
        if (indices.size() != length_)
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");

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

                if (inverse == true) {
                    for (size_t j = 0; j < indices.size(); j++) {
                        const size_t baseIndex = j * indices.size();
                        shiftedState[index] +=
                            conj(matrix[baseIndex + i]) * v[j];
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

    void applyPauliX(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }

    void applyPauliY(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            CFP_t v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                -Util::IMAG<fp_t>() * shiftedState[indices[1]];
            shiftedState[indices[1]] = Util::IMAG<fp_t>() * v0;
        }
    }
    void applyPauliZ(const vector<size_t> &indices,
                     const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= -1;
        }
    }

    void applyHadamard(const vector<size_t> &indices,
                       const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;

            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = Util::INVSQRT2<fp_t>() * (v0 + v1);
            shiftedState[indices[1]] = Util::INVSQRT2<fp_t>() * (v0 - v1);
        }
    }

    void applyS(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift =
            (inverse == true) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    void applyT(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {

        const CFP_t shift = (inverse == true)
                                ? std::conj(std::exp(CFP_t(0, M_PI / 4)))
                                : std::exp(CFP_t(0, M_PI / 4));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    template <typename Param_t = fp_t>
    void applyRX(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);

        const CFP_t js = (inverse == true) ? CFP_t(0, -std::sin(-angle / 2))
                                           : CFP_t(0, std::sin(-angle / 2));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 + js * v1;
            shiftedState[indices[1]] = js * v0 + c * v1;
        }
    }

    template <typename Param_t = fp_t>
    void applyRY(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s = (inverse == true) ? CFP_t(-std::sin(angle / 2), 0)
                                          : CFP_t(std::sin(angle / 2), 0);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }

    template <typename Param_t = fp_t>
    void applyRZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 Param_t angle) {
        const CFP_t first = std::exp(CFP_t(0, -angle / 2));
        const CFP_t second = std::exp(CFP_t(0, angle / 2));
        const CFP_t shift1 = (inverse == true) ? std::conj(first) : first;
        const CFP_t shift2 = (inverse == true) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    template <typename Param_t = fp_t>
    void applyPhaseShift(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         Param_t angle) {
        const CFP_t s = (inverse == true) ? conj(std::exp(CFP_t(0, angle)))
                                          : std::exp(CFP_t(0, angle));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    template <typename Param_t = fp_t>
    void applyRot(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t phi, Param_t theta, Param_t omega) {
        const vector<CFP_t> rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse == true) ? conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse == true) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse == true) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse == true) ? conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }

    void applyCNOT(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }

    void applySWAP(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }
    void applyCZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }

    template <typename Param_t = fp_t>
    void applyCRX(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t js = (inverse == true) ? CFP_t(0, -std::sin(-angle / 2))
                                           : CFP_t(0, std::sin(-angle / 2));

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 + js * v1;
            shiftedState[indices[3]] = js * v0 + c * v1;
        }
    }

    template <typename Param_t = fp_t>
    void applyCRY(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s = (inverse == true) ? CFP_t(-std::sin(angle / 2), 0)
                                          : CFP_t(std::sin(angle / 2), 0);

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    template <typename Param_t = fp_t>
    void applyCRZ(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  Param_t angle) {
        const CFP_t m00 = (inverse == true)
                              ? std::conj(std::exp(CFP_t(0, -angle / 2)))
                              : std::exp(CFP_t(0, -angle / 2));
        const CFP_t m11 = (inverse == true)
                              ? std::conj(std::exp(CFP_t(0, angle / 2)))
                              : std::exp(CFP_t(0, angle / 2));
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    template <typename Param_t = fp_t>
    void applyCRot(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse,
                   Param_t phi, Param_t theta, Param_t omega) {
        const auto rot = Gates::getRot<fp_t>(phi, theta, omega);

        const CFP_t t1 = (inverse == true) ? conj(rot[0]) : rot[0];
        const CFP_t t2 = (inverse == true) ? -rot[1] : rot[1];
        const CFP_t t3 = (inverse == true) ? -rot[2] : rot[2];
        const CFP_t t4 = (inverse == true) ? conj(rot[3]) : rot[3];

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }

    void applyToffoli(const vector<size_t> &indices,
                      const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[6]], shiftedState[indices[7]]);
        }
    }

    void applyCSWAP(const vector<size_t> &indices,
                    const vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[5]], shiftedState[indices[6]]);
        }
    }

    //***********************************************************************//
    //  Internal utility functions for opName dispatch use only.
    //***********************************************************************//
  private:
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

} // namespace Pennylane
