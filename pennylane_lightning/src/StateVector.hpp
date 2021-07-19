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
#include <stdexcept>
#include <vector>

namespace Pennylane {

template <class fp_t = double> class StateVector {
    using CFP_t = std::complex<fp_t>;
    static constexpr CFP_t ONE{1, 0};
    static constexpr CFP_t ZERO{0, 0};
    static constexpr CFP_t IMAG{0, 1};
    static constexpr CFP_t SQRT2{std::sqrt(2), 0};
    static constexpr CFP_t INVSQRT2{1 / std::sqrt(2), 0};

    CFP_t *const arr_;
    const std::size_t length_;

  public:
    StateVector(CFP_t *arr, size_t length) : arr_{arr}, length_{length} {};
    CFP_t *getData() { return arr_; }
    std::size_t getLength() { return length_; }

    static constexpr std::vector<CFP_t> getPauliX() {
        return {ZERO, ONE, ONE, ZERO};
    }
    static constexpr std::vector<CFP_t> getPauliY() {
        return {ZERO, -IMAG, IMAG, ZERO};
    }
    static constexpr std::vector<CFP_t> getPauliZ() {
        return {ONE, ZERO, ZERO, -ONE};
    }
    static constexpr std::vector<CFP_t> getHadamard() {
        return {INVSQRT2, INVSQRT2, INVSQRT2, -INVSQRT2};
    }
    static constexpr std::vector<CFP_t> getS() {
        return {ONE, ZERO, ZERO, IMAG};
    }
    static constexpr std::vector<CFP_t> getT() {
        return {ONE, ZERO, ZERO, IMAG};
    }
    static constexpr std::vector<CFP_t> getCNOT() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ONE,  ZERO};
    }
    static constexpr std::vector<CFP_t> getSWAP() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ONE};
    }
    static constexpr std::vector<CFP_t> getCZ() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, -ONE};
    }

    static constexpr std::vector<CFP_t> getCSWAP() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ONE};
    }
    static constexpr std::vector<CFP_t> getToffoli() {
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO, ZERO, ZERO,
                ZERO, ZERO, ONE,  ZERO};
    }

    static const std::vector<CFP_t> getPhaseShift(fp_t angle) {
        return {ONE, ZERO, ZERO, std::exp(IMAG * angle)};
    }
    static const std::vector<CFP_t> getRX(fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t js(0, -std::sin(angle / 2));
        return {c, js, js, c};
    }
    static const std::vector<CFP_t> getRY(fp_t angle) {
        const CFP_t c(std::cos(angle / 2), 0);
        const CFP_t s(-std::sin(angle / 2), 0);
        return {c, -s, s, c};
    }
    static const std::vector<CFP_t> getRZ(fp_t angle) {
        return {std::exp(-IMAG * (angle / 2)), ZERO, ZERO,
                std::exp(IMAG * (angle / 2))};
    }
    static const std::vector<CFP_t> getRot(fp_t phi, fp_t theta, fp_t omega) {
        const CFP_t c{std::cos(theta / 2), 0}, s{std::sin(theta / 2), 0};
        const fp_t p{phi + omega}, m{phi - omega};
        return std::vector<CFP_t>{
            std::exp(-IMAG * (p / 2)) * c, -std::exp(IMAG * (m / 2)) * s,
            std::exp(-IMAG * (m / 2)) * s, std::exp(IMAG * (p / 2)) * c};
    }
    static const std::vector<CFP_t> getCRX(fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0}, js{0, std::sin(-angle / 2)};
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, c,    js,   ZERO, ZERO, js,   c};
    }
    static const std::vector<CFP_t> getCRY(fp_t angle) {
        const CFP_t c{std::cos(angle / 2), 0}, s{std::sin(angle / 2), 0};
        return {ONE,  ZERO, ZERO, ZERO, ZERO, ONE,  ZERO, ZERO,
                ZERO, ZERO, c,    -s,   ZERO, ZERO, s,    c};
    }
    static const std::vector<CFP_t> getCRZ(fp_t angle) {
        const CFP_t first = std::exp(-IMAG * angle * static_cast<fp_t>(0.5));
        const CFP_t second = std::exp(IMAG * angle * static_cast<fp_t>(0.5));
        return {ONE,  ZERO, ZERO,  ZERO, ZERO, ONE,  ZERO,  ZERO,
                ZERO, ZERO, first, ZERO, ZERO, ZERO, second};
    }
    static const std::vector<CFP_t> getCRot(fp_t phi, fp_t theta, fp_t omega) {
        const auto rot = getRot(phi, theta, omega);
        return {ONE,  ZERO, ZERO,   ZERO,   ZERO, ONE,  ZERO,   ZERO,
                ZERO, ZERO, rot[0], rot[1], ZERO, ZERO, rot[2], rot[3]};
    }
    // Apply Gates
    void applyUnitary(const std::vector<CFP_t> &matrix,
                      const StateVector<CFP_t> &state,
                      const std::vector<size_t> &indices,
                      const std::vector<size_t> &externalIndices,
                      bool inverse) {
        if (indices.size() != length_)
            throw std::out_of_range(
                "The given indices do not match the state-vector length.");

        std::vector<CFP_t> v(indices.size());
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
                        size_t baseIndex = j * indices.size();
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

    void applyPauliX(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }
    void applyPauliY(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            CFP_t v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] = -IMAG * shiftedState[indices[1]];
            shiftedState[indices[1]] = IMAG * v0;
        }
    }
    void applyPauliZ(const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= -1;
        }
    }

    void applyHadamard(const std::vector<size_t> &indices,
                       const std::vector<size_t> &externalIndices,
                       bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;

            const CFP_t v0 = shiftedState[indices[0]];
            const CFP_t v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = INVSQRT2 * (v0 + v1);
            shiftedState[indices[1]] = INVSQRT2 * (v0 - v1);
        }
    }

    void applyS(const std::vector<size_t> &indices,
                const std::vector<size_t> &externalIndices, bool inverse) {
        const CFP_t shift = (inverse == true) ? -IMAG : IMAG;

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    void applyT(const std::vector<size_t> &indices,
                const std::vector<size_t> &externalIndices, bool inverse) {

        const CFP_t shift = (inverse == true)
                                ? std::conj(std::exp(CFP_t{0, M_PI / 4}))
                                : std::exp(CFP_t{0, M_PI / 4});

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }

    void applyRX(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
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

    void applyRY(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
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

    void applyRZ(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse,
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
    void applyPhaseShift(const std::vector<size_t> &indices,
                         const std::vector<size_t> &externalIndices,
                         bool inverse, fp_t angle) {
        const CFP_t s = (inverse == true) ? conj(std::exp(CFP_t{0, angle}))
                                          : std::exp(CFP_t{0, angle});

        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }
    void applyRot(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
                  fp_t phi, fp_t theta, fp_t omega) {
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
    void applyCNOT(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }
    void applySWAP(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }
    void applyCZ(const std::vector<size_t> &indices,
                 const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }
    void applyCRX(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
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
    void applyCRY(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
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
    void applyCRZ(const std::vector<size_t> &indices,
                  const std::vector<size_t> &externalIndices, bool inverse,
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
    void applyCRot(const std::vector<size_t> &indices,
                   const std::vector<size_t> &externalIndices, bool inverse,
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
    void applyToffoli(const std::vector<size_t> &indices,
                      const std::vector<size_t> &externalIndices,
                      bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[6]], shiftedState[indices[7]]);
        }
    }

    void applyCSWAP(const std::vector<size_t> &indices,
                    const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            CFP_t *shiftedState = arr_ + externalIndex;
            std::swap(shiftedState[indices[5]], shiftedState[indices[6]]);
        }
    }
};

} // namespace Pennylane
