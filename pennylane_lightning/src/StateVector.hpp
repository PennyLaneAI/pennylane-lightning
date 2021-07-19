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
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
using std::string;
using std::vector;
}; // namespace

namespace Pennylane {

template <class fp_t = double> class StateVector {
    using CFP_t = std::complex<fp_t>;

    using NPFunc = void (StateVector::*)(const vector<size_t> &,
                                         const vector<size_t> &, bool);
    using PFunc = void (StateVector::*)(const vector<size_t> &,
                                        const vector<size_t> &, bool,
                                        const vector<fp_t> &);

    using NonParamMap = std::unordered_map<string, NPFunc>;
    using ParamMap = std::unordered_map<string, PFunc>;

    static constexpr CFP_t ONE{1, 0};
    static constexpr CFP_t ZERO{0, 0};
    static constexpr CFP_t IMAG{0, 1};
    static constexpr CFP_t SQRT2{std::sqrt(2), 0};
    static constexpr CFP_t INVSQRT2{1 / std::sqrt(2), 0};

    const NonParamMap nonparam_gates_;
    const ParamMap param_gates_;

    CFP_t *const arr_;
    const std::size_t length_;

  public:
    StateVector() : arr_{nullptr}, length_{0} {};
    StateVector(CFP_t *arr, size_t length)
        : arr_{arr}, length_{length},
          nonparam_gates_{{"PauliX", &StateVector<fp_t>::applyPauliX},
                          {"PauliY", &StateVector<fp_t>::applyPauliY},
                          {"PauliZ", &StateVector<fp_t>::applyPauliZ},
                          {"Hadamard", &StateVector<fp_t>::applyHadamard},
                          {"S", &StateVector<fp_t>::applyS},
                          {"T", &StateVector<fp_t>::applyT},
                          {"CNOT", &StateVector<fp_t>::applyCNOT},
                          {"SWAP", &StateVector<fp_t>::applySWAP},
                          {"CSWAP", &StateVector<fp_t>::applyCSWAP},
                          {"CZ", &StateVector<fp_t>::applyCZ},
                          {"Toffoli", &StateVector<fp_t>::applyToffoli}},
          param_gates_{{"PhaseShift", &StateVector<fp_t>::applyPhaseShift},
                       {"RX", &StateVector<fp_t>::applyRX},
                       {"RY", &StateVector<fp_t>::applyRY},
                       {"RZ", &StateVector<fp_t>::applyRZ},
                       {"Rot", &StateVector<fp_t>::applyRot},
                       {"CRX", &StateVector<fp_t>::applyCRX},
                       {"CRY", &StateVector<fp_t>::applyCRY},
                       {"CRZ", &StateVector<fp_t>::applyCRZ},
                       {"CRot", &StateVector<fp_t>::applyCRot}} {};
    CFP_t *getData() { return arr_; }
    std::size_t getLength() { return length_; }

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
    void applyRX(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 const vector<fp_t> &params) {
        return StateVector::applyRX(indices, externalIndices, inverse,
                                    params[0]);
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
    void applyRY(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 const vector<fp_t> &params) {
        return StateVector::applyRY(indices, externalIndices, inverse,
                                    params[0]);
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
    void applyRZ(const vector<size_t> &indices,
                 const vector<size_t> &externalIndices, bool inverse,
                 const vector<fp_t> &params) {
        return StateVector::applyRZ(indices, externalIndices, inverse,
                                    params[0]);
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
    void applyPhaseShift(const vector<size_t> &indices,
                         const vector<size_t> &externalIndices, bool inverse,
                         const vector<fp_t> &params) {
        return StateVector::applyPhaseShift(indices, externalIndices, inverse,
                                            params[0]);
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
    void applyRot(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  const vector<fp_t> &params) {
        return StateVector::applyRot(indices, externalIndices, inverse,
                                     params[0], params[1], params[2]);
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
    void applyCRX(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  const vector<fp_t> &params) {
        return StateVector::applyCRX(indices, externalIndices, inverse,
                                     params[0]);
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
    void applyCRY(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  const vector<fp_t> &params) {
        return StateVector::applyCRY(indices, externalIndices, inverse,
                                     params[0]);
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
    void applyCRZ(const vector<size_t> &indices,
                  const vector<size_t> &externalIndices, bool inverse,
                  const vector<fp_t> &params) {
        return StateVector::applyCRZ(indices, externalIndices, inverse,
                                     params[0]);
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
    void applyCRot(const vector<size_t> &indices,
                   const vector<size_t> &externalIndices, bool inverse,
                   const vector<fp_t> &params) {
        return StateVector::applyCRot(indices, externalIndices, inverse,
                                      params[0], params[1], params[2]);
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
};

} // namespace Pennylane
