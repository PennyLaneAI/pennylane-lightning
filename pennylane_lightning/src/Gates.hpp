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
 * Defines quantum gates and their actions.
 */
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "StateVector.hpp"
#include "Util.hpp"
#include "typedefs.hpp"

namespace {
using namespace Pennylane;

template <class T>
static void validateLength(const std::string &errorPrefix,
                           const std::vector<T> &vec, int requiredLength) {
    if (vec.size() != requiredLength)
        throw std::invalid_argument(
            errorPrefix + ": requires " + std::to_string(requiredLength) +
            " arguments but got " + std::to_string(vec.size()) +
            " arguments instead");
}

template <typename T> class HasApplyKernelMethod {
  private:
    using Found = char[1];
    using NotFound = char[2];

    template <typename U> static Found &f(decltype(&U::applyKernel));
    template <typename U> static NotFound &f(...);

  public:
    enum { value = sizeof(f<T>(0)) == sizeof(Found) };
};
template <typename T> class HasApplyGeneratorMethod {
  private:
    using Found = char[1];
    using NotFound = char[2];

    template <typename U> static Found &f(decltype(&U::applyGenerator));
    template <typename U> static NotFound &f(...);

  public:
    enum { value = sizeof(f<T>(0)) == sizeof(Found) };
};
} // namespace

namespace Pennylane {

template <class Precision> class AbstractGate {
  private:
    const size_t numQubits;
    const size_t length;

  protected:
    AbstractGate(size_t numQubits);

  public:
    /**
     * @return the matrix representation for the gate as a one-dimensional
     * vector.
     */
    virtual const std::vector<std::complex<Precision>> &asMatrix() = 0;

    /**
     * Generic matrix-multiplication kernel
     */
    virtual void applyKernel(const StateVector<Precision> &state,
                             const std::vector<size_t> &indices,
                             const std::vector<size_t> &externalIndices,
                             bool inverse) {

        const std::vector<std::complex<Precision>> &matrix = asMatrix();
        assert(indices.size() == length);

        std::vector<std::complex<Precision>> v(indices.size());
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
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
                    size_t baseIndex = i * indices.size();
                    for (size_t j = 0; j < indices.size(); j++) {
                        shiftedState[index] += matrix[baseIndex + j] * v[j];
                    }
                }
            }
        }
    }

    /**
     * Kernel for applying the generator of an operation
     */
    virtual void applyGenerator(const StateVector<Precision> &state,
                                const std::vector<size_t> &indices,
                                const std::vector<size_t> &externalIndices) {
        throw NotImplementedException();
    };

    /**
     * Scaling factor applied to the generator operation
     */
    static const Precision generatorScalingFactor;
};

// Single-qubit gates:

template <class Precision = double>
class SingleQubitGate : public AbstractGate<Precision> {
  protected:
    SingleQubitGate();
};

template <class Precision = double>
class XGate : public SingleQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{ZERO(), ONE(),
                                                             ONE(), ZERO()};

  public:
    inline static const std::string label = "PauliX";
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    static XGate create(const std::vector<Precision> &parameters) {
        validateLength(label, parameters, 0);
        return XGate<Precision>();
    }

    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            std::swap(shiftedState[indices[0]], shiftedState[indices[1]]);
        }
    }
};

template <class Precision = double>
class YGate : public SingleQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{ZERO(), -IMAG(),
                                                             IMAG(), ZERO()};

  public:
    inline static const std::string label = "PauliY";
    static YGate create(const std::vector<Precision> &parameters) {
        validateLength(label, parameters, 0);
        return Pennylane::YGate();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            std::complex<Precision> v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] =
                -std::complex<Precision>{0, 1} * shiftedState[indices[1]];
            shiftedState[indices[1]] = std::complex<Precision>{0, 1} * v0;
        }
    }
};

template <class Precision = double>
class ZGate : public SingleQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{ONE(), ZERO(),
                                                             ZERO(), -ONE()};

  public:
    inline static const std::string label = "PauliZ";
    static ZGate create(const std::vector<Precision> &parameters) {
        validateLength(label, parameters, 0);
        return ZGate<Precision>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[1]] *= -1;
        }
    }
};

template <class Precision = double>
class HadamardGate : public SingleQubitGate<Precision> {

  public:
    inline static const std::string label = "Hadamard";

    template <class U = Precision>
    static HadamardGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return HadamardGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;

            const std::complex<Precision> v0 = shiftedState[indices[0]];
            const std::complex<Precision> v1 = shiftedState[indices[1]];

            shiftedState[indices[0]] = SQRT2INV * (v0 + v1);
            shiftedState[indices[1]] = SQRT2INV * (v0 - v1);
        }
    }

  private:
    static constexpr std::complex<Precision> SQRT2INV =
        std::complex<Precision>{InvSqrt(2), 0};
    static constexpr std::vector<std::complex<Precision>> matrix{
        SQRT2INV, SQRT2INV, SQRT2INV, -SQRT2INV};
};

template <class Precision = double>
class SGate : public SingleQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{ONE(), ZERO(),
                                                             ZERO(), IMAG()};

  public:
    inline static const std::string label = "S";

    template <class U = Precision>
    static SGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return SGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        std::complex<Precision> shift = (inverse == true) ? -IMAG() : IMAG();

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }
};

template <class Precision = double>
class TGate : public SingleQubitGate<Precision> {
  private:
    static constexpr std::complex<Precision> shift =
        std::pow(M_E, std::complex<Precision>(0, M_PI / 4));
    ;
    static const std::vector<std::complex<Precision>> matrix{ONE(), ZERO(),
                                                             ZERO(), shift};

  public:
    inline static const std::string label = "T";

    template <class U = Precision>
    static TGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return TGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        std::complex<Precision> shift =
            (inverse == true) ? std::conj(shift) : shift;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[1]] *= shift;
        }
    }
};

template <class Precision = double>
class RotationXGate : public SingleQubitGate<Precision> {
  private:
    const std::complex<Precision> c, js;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "RX";

    template <class U = Precision>
    static RotationXGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return RotationXGate<U>(parameters[0]);
    }
    RotationXGate(Precision rotationAngle)
        : c(std::cos(rotationAngle / 2), 0),
          js(0, std::sin(-rotationAngle / 2)){};
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        std::unique_ptr<XGate<Precision>> gate;
        gate->applyKernel(state, indices, externalIndices, false);
    }
    static const Precision generatorScalingFactor = -0.5;
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {

        std::complex<Precision> js_ = (inverse == true) ? -js : js;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            std::complex<Precision> v0 = shiftedState[indices[0]];
            std::complex<Precision> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 + js_ * v1;
            shiftedState[indices[1]] = js_ * v0 + c * v1;
        }
    }
};

template <class Precision = double>
class RotationYGate : public SingleQubitGate<Precision> {
  private:
    const std::complex<Precision> c, s;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "RY";

    template <class U = Precision>
    static RotationYGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return RotationYGate<U>(parameters[0]);
    }
    RotationYGate(Precision rotationAngle)
        : c(std::cos(rotationAngle / 2), 0),
          s(std::sin(rotationAngle / 2), 0), matrix{c, -s, s, c} {}

    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        std::unique_ptr<Pennylane::YGate<Precision>> gate;
        gate->applyKernel(state, indices, externalIndices, false);
    }
    static const Precision generatorScalingFactor{-0.5};
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        const std::complex<Precision> s_ = (inverse == true) ? -s : s;

        for (const size_t &externalIndex : externalIndices) {
            const std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[0]];
            const std::complex<Precision> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = c * v0 - s_ * v1;
            shiftedState[indices[1]] = s_ * v0 + c * v1;
        }
    }
};

template <class Precision = double>
class RotationZGate : public SingleQubitGate<Precision> {
  private:
    const std::complex<Precision> first, second;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "RZ";

    template <class U = Precision>
    static RotationZGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return RotationZGate<U>(parameters[0]);
    }

    RotationZGate(Precision rotationAngle)
        : first(std::pow(M_E, std::complex<Precision>(0, -rotationAngle / 2))),
          second(std::pow(M_E, std::complex<Precision>(0, rotationAngle / 2))),
          matrix(first, ZERO(), ZERO(), second) {}
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {

        std::complex<Precision> shift1 = first;
        std::complex<Precision> shift2 = second;

        if (inverse == true) {
            shift1 = conj(first);
            shift2 = conj(second);
        }

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[0]] *= shift1;
            shiftedState[indices[1]] *= shift2;
        }
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        std::unique_ptr<ZGate<Precision>> gate;
        gate->applyKernel(state, indices, externalIndices, false);
    }
    static const Precision generatorScalingFactor{-0.5};
};

template <class Precision = double>
class PhaseShiftGate : public SingleQubitGate<Precision> {
  private:
    const std::complex<Precision> shift;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "PhaseShift";

    template <class U = Precision>
    static PhaseShiftGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return PhaseShiftGate<U>(parameters[0]);
    }
    PhaseShiftGate(Precision rotationAngle)
        : shift(std::pow(M_E, std::complex(0, rotationAngle))),
          matrix(ONE(), ZERO(), ZERO(), shift) {}
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {

        std::complex<Precision> s = (inverse == true) ? conj(shift) : shift;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState = state.arr + externalIndex;
            shiftedState[indices[1]] *= s;
        }
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        for (const size_t &externalIndex : externalIndices) {
            const std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[0]] = 0;
        }
    }
    static const Precision generatorScalingFactor{1.0};
};

template <class Precision = double>
class GeneralRotationGate : public SingleQubitGate<Precision> {
  private:
    const std::complex<Precision> c, s, r1, r2, r3, r4;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "Rot";

    template <class U = Precision>
    static GeneralRotationGate<U> create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 3);
        return GeneralRotationGate<U>(parameters[0], parameters[1],
                                      parameters[2]);
    }
    GeneralRotationGate(Precision phi, Precision theta, Precision omega)
        : c(std::cos(theta / 2), 0), s(std::sin(theta / 2), 0),
          r1(c * std::pow(M_E, std::complex<Precision>(0, (-phi - omega) / 2))),
          r2(-s * std::pow(M_E, std::complex<Precision>(0, (phi - omega) / 2))),
          r3(s * std::pow(M_E, std::complex<Precision>(0, (-phi + omega) / 2))),
          r4(c * std::pow(M_E, std::complex<Precision>(0, (phi + omega) / 2))),
          matrix(r1, r2, r3, r4) {}
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        std::complex<Precision> t1 = r1;
        std::complex<Precision> t2 = r2;
        std::complex<Precision> t3 = r3;
        std::complex<Precision> t4 = r4;

        if (inverse == true) {
            t1 = conj(r1);
            t2 *= -1;
            t3 *= -1;
            t4 = conj(t4);
        }

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[0]];
            const std::complex<Precision> v1 = shiftedState[indices[1]];
            shiftedState[indices[0]] = t1 * v0 + t2 * v1;
            shiftedState[indices[1]] = t3 * v0 + t4 * v1;
        }
    }
};

// Two-qubit gates
template <class Precision = double>
class TwoQubitGate : public AbstractGate<TwoQubitGate<Precision>> {
  private:
    using precision_ = Precision;

  protected:
    TwoQubitGate();
};

template <class Precision = double>
class CNOTGate : public TwoQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{
        ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ONE(),  ZERO()};

  public:
    inline static const std::string label = "CNOT";

    template <class U = Precision>
    static CNOTGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return CNOTGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            std::swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }
};

template <class Precision = double>
class SWAPGate : public TwoQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{
        ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(),
        ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE()};

  public:
    inline static const std::string label = "SWAP";

    template <class U = Precision>
    static SWAPGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return SWAPGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            swap(shiftedState[indices[1]], shiftedState[indices[2]]);
        }
    }
};

template <class Precision = double>
class CZGate : public TwoQubitGate<Precision> {
  private:
    static constexpr std::vector<std::complex<Precision>> matrix{
        ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(),
        ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), -ONE()};

  public:
    inline static const std::string label = "CZ";

    template <class U = Precision>
    static CZGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return CZGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[3]] *= -1;
        }
    }
};

template <class Precision = double>
class CRotationXGate : public TwoQubitGate<Precision> {
  private:
    const std::complex<Precision> c, js;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "CRX";

    template <class U = Precision>
    static CRotationXGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return CRotationXGate<U>(parameters[0]);
    }
    CRotationXGate(Precision rotationAngle)
        : c(std::cos(rotationAngle / 2), 0),
          js(0, std::sin(-rotationAngle / 2)),
          matrix(ONE(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(), ZERO(), ZERO(),
                 ZERO(), ZERO(), c, js, ZERO(), ZERO(), js, c){};
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        std::complex<Precision> js_ = (inverse == true) ? -js : js;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[2]];
            const std::complex<Precision> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 + js_ * v1;
            shiftedState[indices[3]] = js_ * v0 + c * v1;
        }
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
            swap(shiftedState[indices[2]], shiftedState[indices[3]]);
        }
    }
    static const Precision generatorScalingFactor{-0.5};
};

template <class Precision = double>
class CRotationYGate : public TwoQubitGate<Precision> {
  private:
    const std::complex<Precision> c, s;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "CRY";

    template <class U = Precision>
    static CRotationYGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return CRotationYGate<U>(parameters[0]);
    }

    CRotationYGate(Precision rotationAngle)
        : c(std::cos(rotationAngle / 2), 0), s(std::sin(rotationAngle / 2), 0),
          matrix(ONE(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(), ZERO(), ZERO(),
                 ZERO(), ZERO(), c, -s, ZERO(), ZERO(), s, c) {}

    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        const std::complex<Precision> s_ = (inverse == true) ? -s : s;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[2]];
            const std::complex<Precision> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = c * v0 - s_ * v1;
            shiftedState[indices[3]] = s_ * v0 + c * v1;
        }
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[0]];
            shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
            shiftedState[indices[2]] =
                -IMAG<Precision>() * shiftedState[indices[3]];
            shiftedState[indices[3]] = IMAG<Precision>() * v0;
        }
    }
    static const Precision generatorScalingFactor{0.5};
};

template <class Precision = double>
class CRotationZGate : public TwoQubitGate<Precision> {
  private:
    const std::complex<Precision> first, second;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "CRZ";

    template <class U = Precision>
    static CRotationZGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 1);
        return CRotationZGate<U>(parameters[0]);
    }
    CRotationZGate(Precision rotationAngle)
        : first(std::pow(M_E, std::complex<Precision>(0, -rotationAngle / 2))),
          second(std::pow(M_E, std::complex<Precision>(0, rotationAngle / 2))),
          matrix(ONE(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(), ZERO(), ZERO(),
                 ZERO(), ZERO(), first, ZERO(), ZERO(), ZERO(), ZERO(),
                 second) {}
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        const std::complex<Precision> shift1 = (inverse) ? conj(first) : first;
        const std::complex<Precision> shift2 =
            (inverse) ? conj(second) : second;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[2]] *= shift1;
            shiftedState[indices[3]] *= shift2;
        }
    }
    void applyGenerator(const StateVector<Precision> &state,
                        const std::vector<size_t> &indices,
                        const std::vector<size_t> &externalIndices) {
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
            shiftedState[indices[3]] *= -1;
        }
    }
    static const Precision generatorScalingFactor{-0.5};
};

template <class Precision = double>
class CGeneralRotationGate : public TwoQubitGate<Precision> {
  private:
    const std::complex<Precision> c, s, r1, r2, r3, r4;
    const std::vector<std::complex<Precision>> matrix;

  public:
    inline static const std::string label = "CRot";

    template <class U = Precision>
    static CGeneralRotationGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 3);
        return CGeneralRotationGate<U>(parameters[0], parameters[1],
                                       parameters[2]);
    }
    CGeneralRotationGate(Precision phi, Precision theta, Precision omega)
        : c(std::cos(theta / 2), 0), s(std::sin(theta / 2), 0),
          r1(c * std::pow(M_E, std::complex<Precision>(0, (-phi - omega) / 2))),
          r2(-s * std::pow(M_E, std::complex<Precision>(0, (phi - omega) / 2))),
          r3(s * std::pow(M_E, std::complex<Precision>(0, (-phi + omega) / 2))),
          r4(c * std::pow(M_E, std::complex<Precision>(0, (phi + omega) / 2))),
          matrix{ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(),
                 ZERO(), ZERO(), r1,     r2,     ZERO(), ZERO(), r3,     r4} {}

    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {

        const std::complex<Precision> t1 = (inverse) ? conj(r1) : r1;
        const std::complex<Precision> t2 = (inverse) ? -r2 : r2;
        const std::complex<Precision> t3 = (inverse) ? -r3 : r3;
        const std::complex<Precision> t4 = (inverse) ? conj(r4) : r4;

        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            const std::complex<Precision> v0 = shiftedState[indices[2]];
            const std::complex<Precision> v1 = shiftedState[indices[3]];
            shiftedState[indices[2]] = t1 * v0 + t2 * v1;
            shiftedState[indices[3]] = t3 * v0 + t4 * v1;
        }
    }
};

// Three-qubit gates

template <class Precision = double>
class ThreeQubitGate : public AbstractGate<ThreeQubitGate<Precision>> {
  private:
    using precision_ = Precision;

  protected:
    ThreeQubitGate();
};

template <class Precision = double>
class ToffoliGate : public ThreeQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{
        ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO()};

  public:
    inline static const std::string label = "Toffoli";

    template <class U = Precision>
    static ToffoliGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return ToffoliGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            swap(shiftedState[indices[6]], shiftedState[indices[7]]);
        }
    }
};

template <class Precision = double>
class CSWAPGate : public ThreeQubitGate<Precision> {
  private:
    static const std::vector<std::complex<Precision>> matrix{
        ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE(),  ZERO(), ZERO(),
        ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ZERO(), ONE()};

  public:
    inline static const std::string label = "CSWAP";

    template <class U = Precision>
    static CSWAPGate create(const std::vector<U> &parameters) {
        validateLength(label, parameters, 0);
        return CSWAPGate<U>();
    }
    inline const std::vector<std::complex<Precision>> &asMatrix() {
        return matrix;
    }
    void applyKernel(const StateVector<Precision> &state,
                     const std::vector<size_t> &indices,
                     const std::vector<size_t> &externalIndices, bool inverse) {
        // gate is its own inverse
        for (const size_t &externalIndex : externalIndices) {
            std::complex<Precision> *shiftedState =
                state.getData() + externalIndex;
            swap(shiftedState[indices[5]], shiftedState[indices[6]]);
        }
    }
};

// Use anon namespace to avoid exposing unnecessary functions
namespace {

static const auto dispatchTable = createDispatchTable();

template <class GateType, class Precision>
static void addToDispatchTable(
    std::unordered_map<
        std::string,
        std::function<std::unique_ptr<Pennylane::AbstractGate<Precision>>(
            const std::vector<Precision> &)>> &dispatchTable) {
    dispatchTable.emplace(
        GateType::label, [](const std::vector<Precision> &parameters) {
            return make_unique<GateType>(GateType::create(parameters));
        });
}

template <class Precision = double> static auto createDispatchTable() {
    std::unordered_map<
        std::string,
        std::function<std::unique_ptr<Pennylane::AbstractGate<Precision>>(
            const std::vector<Precision> &)>>
        dispatchTable;
    addToDispatchTable<Pennylane::XGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::YGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::ZGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::HadamardGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::SGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::TGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::RotationXGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::RotationYGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::RotationZGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::PhaseShiftGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::GeneralRotationGate<Precision>>(
        dispatchTable);
    addToDispatchTable<Pennylane::CNOTGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::SWAPGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CZGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationXGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationYGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationZGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CGeneralRotationGate<Precision>>(
        dispatchTable);
    addToDispatchTable<Pennylane::ToffoliGate<Precision>>(dispatchTable);
    addToDispatchTable<Pennylane::CSWAPGate<Precision>>(dispatchTable);
    return dispatchTable;
}

static const auto dispatchTableD = createDispatchTable();
} // namespace

/**
 * Produces the requested gate, defined by a label and the list of parameters
 *
 * @param label unique string corresponding to a gate type
 * @param parameters defines the gate parameterisation (may be zero-length for
 * some gates)
 * @return the gate wrapped in std::unique_ptr
 * @throws std::invalid_argument thrown if the gate type is not defined, or if
 * the number of parameters to the gate is incorrect
 */
template <class Precision>
std::unique_ptr<AbstractGate<Precision>>
constructGate(const std::string &label,
              const std::vector<std::complex<Precision>> &parameters) {
    auto dispatchTableIterator = dispatchTable.find(label);
    if (dispatchTableIterator == dispatchTable.end())
        throw std::invalid_argument(label + " is not a supported gate type");

    return dispatchTableIterator->second(parameters);
}

} // namespace Pennylane
