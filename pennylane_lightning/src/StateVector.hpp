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

#include <cmath>
#include <complex>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Util.hpp"

namespace {
using namespace std::placeholders;

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

    static constexpr CFP_t ONE = {1, 0};
    static constexpr CFP_t ZERO = {0, 0};
    static constexpr CFP_t IMAG = {0, 1};
    inline static const CFP_t SQRT2 = {static_cast<fp_t>(std::sqrt(2)), 0};
    inline static const CFP_t INVSQRT2 = {static_cast<fp_t>(1 / std::sqrt(2)),
                                          0};

    const FMap gates_;
    const std::unordered_map<string, size_t> gate_wires_;

    CFP_t *const arr_;
    const size_t length_;
    const size_t num_qubits_;

    vector<size_t>
    getIndicesAfterExclusion(const vector<size_t> &indicesToExclude,
                             size_t qubits) {
        std::set<size_t> indices;
        for (size_t i = 0; i < qubits; i++) {
            indices.emplace(i);
        }
        for (const size_t &excludedIndex : indicesToExclude) {
            indices.erase(excludedIndex);
        }
        return {indices.begin(), indices.end()};
    }

    vector<size_t> generateBitPatterns(const vector<size_t> &qubitIndices,
                                       size_t qubits) {
        vector<size_t> indices;
        indices.reserve(Util::exp2(qubitIndices.size()));
        indices.emplace_back(0);
        for (int i = qubitIndices.size() - 1; i >= 0; i--) {
            size_t value = Util::maxDecimalForQubit(qubitIndices[i], qubits);
            size_t currentSize = indices.size();
            for (size_t j = 0; j < currentSize; j++) {
                indices.emplace_back(indices[j] + value);
            }
        }
        return indices;
    }

  public:
    StateVector() : arr_{nullptr}, length_{0}, num_qubits_{0} {};
    StateVector(CFP_t *arr, size_t length)
        : arr_{arr}, length_{length}, num_qubits_{Util::fast_log2(length_)},
          gate_wires_{
              {"PauliX", 1}, {"PauliY", 1}, {"PauliZ", 1},     {"Hadamard", 1},
              {"T", 1},      {"S", 1},      {"RX", 1},         {"RY", 1},
              {"RZ", 1},     {"Rot", 1},    {"PhaseShift", 1}, {"CNOT", 2},
              {"SWAP", 2},   {"CZ", 2},     {"CRX", 2},        {"CRY", 2},
              {"CRZ", 2},    {"CRot", 2},   {"CSWAP", 3},      {"Toffoli", 3}},
          gates_{
              {"PauliX",
               std::bind(&StateVector::applyPauliX_, this, _1, _2, _3, _4)},
              {"PauliY",
               std::bind(&StateVector::applyPauliY_, this, _1, _2, _3, _4)},
              {"PauliZ",
               std::bind(&StateVector::applyPauliZ_, this, _1, _2, _3, _4)},
              {"Hadamard",
               std::bind(&StateVector::applyHadamard_, this, _1, _2, _3, _4)},
              {"S", std::bind(&StateVector::applyS_, this, _1, _2, _3, _4)},
              {"T", std::bind(&StateVector::applyT_, this, _1, _2, _3, _4)},
              {"CNOT",
               std::bind(&StateVector::applyCNOT_, this, _1, _2, _3, _4)},
              {"SWAP",
               std::bind(&StateVector::applySWAP_, this, _1, _2, _3, _4)},
              {"CSWAP",
               std::bind(&StateVector::applyCSWAP_, this, _1, _2, _3, _4)},
              {"CZ", std::bind(&StateVector::applyCZ_, this, _1, _2, _3, _4)},
              {"Toffoli",
               std::bind(&StateVector::applyToffoli_, this, _1, _2, _3, _4)},
              {"PhaseShift",
               std::bind(&StateVector::applyPhaseShift_, this, _1, _2, _3, _4)},
              {"RX", std::bind(&StateVector::applyRX_, this, _1, _2, _3, _4)},
              {"RY", std::bind(&StateVector::applyRY_, this, _1, _2, _3, _4)},
              {"RZ", std::bind(&StateVector::applyRZ_, this, _1, _2, _3, _4)},
              {"Rot", std::bind(&StateVector::applyRot_, this, _1, _2, _3, _4)},
              {"CRX", std::bind(&StateVector::applyCRX_, this, _1, _2, _3, _4)},
              {"CRY", std::bind(&StateVector::applyCRY_, this, _1, _2, _3, _4)},
              {"CRZ", std::bind(&StateVector::applyCRZ_, this, _1, _2, _3, _4)},
              {"CRot",
               std::bind(&StateVector::applyCRot_, this, _1, _2, _3, _4)}} {};

    CFP_t *getData() { return arr_; }
    std::size_t getLength() { return length_; }

    void apply(const string &opName, const vector<size_t> &wires,
               const vector<fp_t> &params, bool inverse) {
        const auto gate = gates_.at(opName);
        if (gate_wires_.at(opName) != wires.size())
            throw std::invalid_argument(
                string("The gate of type ") + opName + " requires " +
                std::to_string(gate_wires_.at(opName)) + " wires, but " +
                std::to_string(wires.size()) + " were supplied");

        // assume copy elision
        const vector<size_t> internalIndices =
            generateBitPatterns(wires, num_qubits_);
        const vector<size_t> externalWires =
            getIndicesAfterExclusion(wires, num_qubits_);
        const vector<size_t> externalIndices =
            generateBitPatterns(externalWires, num_qubits_);

        // To be relplaced with variadic function; additionally, SFINAE may
        // assist
        /*if (params.size() > 0)
            applyParametric(opName, internalIndices, externalIndices, inverse,
                            params);
        else
            applyNonParametric(opName, internalIndices, externalIndices,
                               inverse);*/
    }
    /*
        void apply(const string &opName, const vector<size_t> &wires,
                   const vector<fp_t> &params, bool inverse) {
            if (num_qubits_ != wires.size())
                throw std::invalid_argument(
                    string("The gate of type ") + opName + " requires " +
                    std::to_string(num_qubits_) + " wires, but " +
                    std::to_string(wires.size()) + " were supplied");

            // assume copy elision
            const vector<size_t> internalIndices =
                generateBitPatterns(wires, num_qubits_);
            const vector<size_t> externalWires =
                getIndicesAfterExclusion(wires, num_qubits_);
            const vector<size_t> externalIndices =
                generateBitPatterns(externalWires, num_qubits_);

            // To be relplaced with variadic function; additionally, SFINAE may
            // assist
            if (params.size() > 0)
                applyParametric(opName, internalIndices, externalIndices,
       inverse, params); else applyNonParametric(opName, internalIndices,
       externalIndices, inverse);
        }

        void applyNonParametric(const string &opName,
                                const vector<size_t> &internalIndices,
                                const vector<size_t> &externalIndices,
                                bool inverse) {
            const auto fptr = nonparam_gates_.at(opName);
            fptr(internalIndices, externalIndices, inverse);
        }
        void applyParametric(const string &opName,
                             const vector<size_t> &internalIndices,
                             const vector<size_t> &externalIndices, bool
       inverse, const vector<fp_t> &params) { const auto fptr =
       param_gates_.at(opName); fptr(internalIndices, externalIndices, inverse,
       params);
        }*/

    static constexpr vector<CFP_t> getPauliX() {
        return {ZERO, ONE, ONE, ZERO};
    }
    static constexpr vector<CFP_t> getPauliY() {
        return {ZERO, -IMAG, IMAG, ZERO};
    }
    static constexpr vector<CFP_t> getPauliZ() {
        return {ONE, ZERO, ZERO, -ONE};
    }
    static constexpr vector<CFP_t> getHadamard() {
        return {INVSQRT2, INVSQRT2, INVSQRT2, -INVSQRT2};
    }
    static constexpr vector<CFP_t> getS() { return {ONE, ZERO, ZERO, IMAG}; }
    static constexpr vector<CFP_t> getT() { return {ONE, ZERO, ZERO, IMAG}; }
    static constexpr vector<CFP_t> getCNOT() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ONE,  ZERO};
    }
    static constexpr vector<CFP_t> getSWAP() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ONE};
    }
    static constexpr vector<CFP_t> getCZ() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, -ONE};
    }

    static constexpr vector<CFP_t> getCSWAP() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE};
    }
    static constexpr vector<CFP_t> getToffoli() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ONE,  ZERO};
    }

    static const vector<CFP_t> getPhaseShift(fp_t angle) {
        return {ONE, ZERO, ZERO, std::exp(IMAG * angle)};
    }
    static const vector<CFP_t> getPhaseShift(const vector<fp_t> &params) {
        return StateVector::getPhaseShift(params.front());
    }
    static const vector<CFP_t> getRX(fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t js(0, -std::sin(angle / 2));
        return {c, js, js, c};
    }
    static const vector<CFP_t> getRX(const vector<fp_t> &params) {
        return StateVector::getRX(params.front());
    }
    static const vector<CFP_t> getRY(fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s(-std::sin(angle / 2), 0);
        return {c, -s, s, c};
    }
    static const vector<CFP_t> getRY(const vector<fp_t> &params) {
        return StateVector::getRY(params.front());
    }
    static const vector<CFP_t> getRZ(fp_t angle) {
        return {std::exp(-IMAG * (angle / 2)), ZERO, ZERO,
                std::exp(IMAG * (angle / 2))};
    }
    static const vector<CFP_t> getRZ(const vector<fp_t> &params) {
        return StateVector::getRZ(params.front());
    }

    static const vector<CFP_t> getRot(fp_t phi, fp_t theta, fp_t omega) {
        const CFP_t c{std::cos(theta / 2), 0}, s{std::sin(theta / 2), 0};
        const fp_t p{phi + omega}, m{phi - omega};
        return vector<CFP_t>{
            std::exp(-IMAG * (p / 2)) * c, -std::exp(IMAG * (m / 2)) * s,
            std::exp(-IMAG * (m / 2)) * s, std::exp(IMAG * (p / 2)) * c};
    }
    static const vector<CFP_t> getRot(const vector<fp_t> &params) {
        return StateVector::getRot(params[0], params[1], params[2]);
    }

    static const vector<CFP_t> getCRX(fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0}, js{0, std::sin(-angle / 2)};
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, c,    js,   ZERO, ZERO, js,   c};
    }
    static const vector<CFP_t> getCRX(const vector<fp_t> &params) {
        return StateVector::getCRX(params.front());
    }
    static const vector<CFP_t> getCRY(fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0}, s{std::sin(angle / 2), 0};
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, c,    -s,   ZERO, ZERO, s,    c};
    }
    static const vector<CFP_t> getCRY(const vector<fp_t> &params) {
        return StateVector::getCRY(params.front());
    }
    static const vector<CFP_t> getCRZ(fp_t angle) {
        const CFP_t first = std::exp(-IMAG * angle * static_cast<fp_t>(0.5));
        const CFP_t second = std::exp(IMAG * angle * static_cast<fp_t>(0.5));
        return {ONE,  ZERO, ZERO,  ZERO, ZERO, ONE,  ZERO,  ZERO,
                ZERO, ZERO, first, ZERO, ZERO, ZERO, second};
    }
    static const vector<CFP_t> getCRZ(const vector<fp_t> &params) {
        return StateVector::getCRZ(params.front());
    }

    static const vector<CFP_t> getCRot(fp_t phi, fp_t theta, fp_t omega) {
        const auto rot = getRot(phi, theta, omega);
        return {ONE,  ZERO, ZERO,   ZERO,   ZERO, ONE,  ZERO,   ZERO,
                ZERO, ZERO, rot[0], rot[1], ZERO, ZERO, rot[2], rot[3]};
    }
    static const vector<CFP_t> getCRot(const vector<fp_t> &params) {
        return StateVector::getCRot(params[0], params[1], params[2]);
    }

    // Apply Gates
    void applyUnitary(const vector<CFP_t> &matrix,
                      const StateVector<CFP_t> &state,
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
            shiftedState[indices[0]] = -IMAG * shiftedState[indices[1]];
            shiftedState[indices[1]] = IMAG * v0;
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

            shiftedState[indices[0]] = INVSQRT2 * (v0 + v1);
            shiftedState[indices[1]] = INVSQRT2 * (v0 - v1);
        }
    }

    void applyS(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift = (inverse == true) ? -IMAG : IMAG;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    void applyT(const vector<size_t> &indices,
                const vector<size_t> &externalIndices, bool inverse) {

        const CFP_t shift = (inverse == true)
                                ? std::conj(std::exp(CFP_t{0, M_PI / 4}))
                                : std::exp(CFP_t{0, M_PI / 4});

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    void applyRX(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);

        const CFP_t js = (inverse == true) ? CFP_t{0, -std::sin(-angle / 2)}
                                           : CFP_t{0, std::sin(-angle / 2)};

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 + js * v1;
            shiftedState[indices[1]] = js * v0 + c * v1;
        }
    }

    void applyRY(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s = (inverse == true) ? CFP_t{-std::sin(angle / 2), 0}
                                          : CFP_t{std::sin(angle / 2), 0};

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s * v1;
            shiftedState[indices[1]] = s * v0 + c * v1;
        }
    }

    void applyRZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 fp_t angle) {

        const CFP_t first = std::exp(CFP_t{0, -angle / 2});
        const CFP_t second = std::exp(CFP_t{0, angle / 2});
        const CFP_t shift1 = (inverse == true) ? std::conj(first) : first;
        const CFP_t shift2 = (inverse == true) ? std::conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }

    void applyPhaseShift(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         fp_t angle) {
        const CFP_t s = (inverse == true) ? conj(std::exp(CFP_t{0, angle}))
                                          : std::exp(CFP_t{0, angle});

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }

    void applyRot(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse, fp_t phi,
                  fp_t theta, fp_t omega) {
        const auto rot = getRot(phi, theta, omega);

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
    void applyCRX(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0};
        const CFP_t js = (inverse == true) ? CFP_t{0, -std::sin(-angle / 2)}
                                           : CFP_t{0, std::sin(-angle / 2)};

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 + js * v1;
            shiftedState[indices[3]] = js * v0 + c * v1;
        }
    }

    void applyCRY(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0};
        const CFP_t s = (inverse == true) ? CFP_t{-std::sin(angle / 2), 0}
                                          : CFP_t{std::sin(angle / 2), 0};

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            const CFP_t v0 = shiftedState[indices[2]];
            const CFP_t v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s * v1;
            shiftedState[indices[3]] = s * v0 + c * v1;
        }
    }

    void applyCRZ(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  fp_t angle) {
        const CFP_t m00 = (inverse == true)
                              ? std::conj(std::exp(CFP_t{0, -angle / 2}))
                              : std::exp(CFP_t{0, -angle / 2});
        const CFP_t m11 = (inverse == true)
                              ? std::conj(std::exp(CFP_t{0, angle / 2}))
                              : std::exp(CFP_t{0, angle / 2});
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[2]] *= m00;
            shiftedState[indices[3]] *= m11;
        }
    }

    void applyCRot(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse,
                   fp_t phi, fp_t theta, fp_t omega) {
        const auto rot = getRot(phi, theta, omega);

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
        applyPauliX(indices, externalIndices, inverse);
    }
    inline void applyPauliY_(const vector<size_t> &indices,
                             const vector<size_t> &externalIndices,
                             bool inverse, const vector<fp_t> &params) {
        applyPauliY(indices, externalIndices, inverse);
    }
    inline void applyPauliZ_(const vector<size_t> &indices,
                             const vector<size_t> &externalIndices,
                             bool inverse, const vector<fp_t> &params) {
        applyPauliZ(indices, externalIndices, inverse);
    }
    inline void applyHadamard_(const vector<size_t> &indices,
                               const vector<size_t> &externalIndices,
                               bool inverse, const vector<fp_t> &params) {
        applyHadamard(indices, externalIndices, inverse);
    }
    inline void applyS_(const vector<size_t> &indices,
                        const vector<size_t> &externalIndices, bool inverse,
                        const vector<fp_t> &params) {
        applyS(indices, externalIndices, inverse);
    }
    inline void applyT_(const vector<size_t> &indices,
                        const vector<size_t> &externalIndices, bool inverse,
                        const vector<fp_t> &params) {
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
        applyCNOT(indices, externalIndices, inverse);
    }
    inline void applySWAP_(const vector<size_t> &indices,
                           const vector<size_t> &externalIndices, bool inverse,
                           const vector<fp_t> &params) {
        applySWAP(indices, externalIndices, inverse);
    }
    inline void applyCZ_(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
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
        applyToffoli(indices, externalIndices, inverse);
    }
    inline void applyCSWAP_(const vector<size_t> &indices,
                            const vector<size_t> &externalIndices, bool inverse,
                            const vector<fp_t> &params) {
        applyCSWAP(indices, externalIndices, inverse);
    }
};

} // namespace Pennylane
