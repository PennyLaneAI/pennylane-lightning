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

#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <iostream>

#include "Gates.hpp"
#include "Util.hpp"

using std::function;
using std::map;
using std::string;
using std::swap;
using std::unique_ptr;
using std::vector;
using std::conj;

using Pennylane::CplxType;

template<class T>
static void validateLength(const string& errorPrefix, const vector<T>& vec, int requiredLength) {
    if (vec.size() != requiredLength)
        throw std::invalid_argument(errorPrefix + ": requires " + std::to_string(requiredLength) + " arguments but got " + std::to_string(vec.size()) + " arguments instead");
}

// -------------------------------------------------------------------------------------------------------------

Pennylane::AbstractGate::AbstractGate(int numQubits)
    : numQubits(numQubits)
    , length(exp2(numQubits))
{}

void Pennylane::AbstractGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    const vector<CplxType>& matrix = asMatrix();
    auto len_indices = indices.size();
    assert(len_indices == length);

    vector<CplxType> v(len_indices);
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        // Gather
        size_t pos = 0;
        for (const size_t& index : indices) {
            v[pos] = shiftedState[index];
            pos++;
        }

        // Apply + scatter
        for (size_t i = 0; i < len_indices; i++) {
            size_t index = indices[i];
            shiftedState[index] = 0;

            if (inverse == true) {
                for (size_t j = 0; j < len_indices; j++) {
                    size_t baseIndex = j * len_indices;
                    shiftedState[index] += conj(matrix[baseIndex + i]) * v[j];
                }
            } else {
                size_t baseIndex = i * len_indices;
                for (size_t j = 0; j < len_indices; j++) {
                    shiftedState[index] += matrix[baseIndex + j] * v[j];
                }
            }
        }
    }
}

const double Pennylane::AbstractGate::generatorScalingFactor{};
void Pennylane::AbstractGate::applyGenerator(const StateVector&, const std::vector<size_t>&, const std::vector<size_t>&) {
    throw NotImplementedException();
}

// -------------------------------------------------------------------------------------------------------------

Pennylane::SingleQubitGate::SingleQubitGate()
    : AbstractGate(1)
{}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::XGate::label = "PauliX";

Pennylane::XGate::XGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)

{}

Pennylane::XGate Pennylane::XGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::XGate::label, parameters, 0);
    return Pennylane::XGate(wires);
}

const vector<CplxType> Pennylane::XGate::matrix{
    0, 1,
    1, 0 };

void Pennylane::XGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        swap(shiftedState[indices[0]], shiftedState[indices[1]]);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::YGate::label = "PauliY";

Pennylane::YGate Pennylane::YGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::YGate::label, parameters, 0);
    return Pennylane::YGate(wires);
}

const vector<CplxType> Pennylane::YGate::matrix{
    0, -IMAG,
    IMAG, 0 };

Pennylane::YGate::YGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)

{}

void Pennylane::YGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        shiftedState[indices[0]] = -IMAG * shiftedState[indices[1]];
        shiftedState[indices[1]] = IMAG * v0;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::ZGate::label = "PauliZ";

Pennylane::ZGate Pennylane::ZGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::ZGate::label, parameters, 0);
    return Pennylane::ZGate(wires);
}

const std::vector<CplxType> Pennylane::ZGate::matrix{
    1, 0,
    0, -1 };

Pennylane::ZGate::ZGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires(vector<unsigned int>{})
    , targetWires(wires)

{}

void Pennylane::ZGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[1]] *= -1;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::HadamardGate::label = "Hadamard";

Pennylane::HadamardGate Pennylane::HadamardGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::HadamardGate::label, parameters, 0);
    return Pennylane::HadamardGate(wires);
}

const vector<CplxType> Pennylane::HadamardGate::matrix{
    SQRT2INV, SQRT2INV,
    SQRT2INV, -SQRT2INV };

Pennylane::HadamardGate::HadamardGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)

{}

void Pennylane::HadamardGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        CplxType v1 = shiftedState[indices[1]];
        shiftedState[indices[0]] = SQRT2INV * (v0 + v1);
        shiftedState[indices[1]] = SQRT2INV * (v0 - v1);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::SGate::label = "S";

Pennylane::SGate Pennylane::SGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::SGate::label, parameters, 0);
    return Pennylane::SGate(wires);
}

const vector<CplxType> Pennylane::SGate::matrix{
    1, 0,
    0, IMAG };

Pennylane::SGate::SGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)

{}

void Pennylane::SGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType shift = (inverse == true) ? -IMAG : IMAG;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[1]] *= shift;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::TGate::label = "T";

Pennylane::TGate Pennylane::TGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::TGate::label, parameters, 0);
    return Pennylane::TGate(wires);
}

const CplxType Pennylane::TGate::shift = std::pow(M_E, CplxType(0, M_PI / 4));

const vector<CplxType> Pennylane::TGate::matrix{
    1, 0,
    0, Pennylane::TGate::shift };

Pennylane::TGate::TGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)

{}

void Pennylane::TGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType shift = (inverse == true) ? conj(Pennylane::TGate::shift) : Pennylane::TGate::shift;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[1]] *= shift;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::RotationXGate::label = "RX";

Pennylane::RotationXGate Pennylane::RotationXGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::RotationXGate::label, parameters, 1);
    return Pennylane::RotationXGate(parameters[0], wires);
}

Pennylane::RotationXGate::RotationXGate(double rotationAngle, const vector<unsigned int>& wires)
    : c(std::cos(rotationAngle / 2), 0)
    , js(0, std::sin(-rotationAngle / 2))
    , matrix{
      c, js,
      js, c }
    , allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

const double Pennylane::RotationXGate::generatorScalingFactor{ -0.5 };

void Pennylane::RotationXGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    unique_ptr<Pennylane::XGate> gate;
    gate->applyKernel(state, indices, externalIndices, false);
}

void Pennylane::RotationXGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType js_ = (inverse == true) ? -js : js;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        CplxType v1 = shiftedState[indices[1]];
        shiftedState[indices[0]] = c * v0 + js_ * v1;
        shiftedState[indices[1]] = js_ * v0 + c * v1;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::RotationYGate::label = "RY";

Pennylane::RotationYGate Pennylane::RotationYGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::RotationYGate::label, parameters, 1);
    return Pennylane::RotationYGate(parameters[0], wires);
}

Pennylane::RotationYGate::RotationYGate(double rotationAngle, const vector<unsigned int>& wires)
    : c(std::cos(rotationAngle / 2), 0)
    , s(std::sin(rotationAngle / 2), 0)
    , matrix{
      c, -s,
      s, c }
    , allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

const double Pennylane::RotationYGate::generatorScalingFactor{ -0.5 };

void Pennylane::RotationYGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    unique_ptr<Pennylane::YGate> gate;
    gate->applyKernel(state, indices, externalIndices, false);
}

void Pennylane::RotationYGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType s_ = (inverse == true) ? -s : s;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        CplxType v1 = shiftedState[indices[1]];
        shiftedState[indices[0]] = c * v0 - s_ * v1;
        shiftedState[indices[1]] = s_ * v0 + c * v1;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::RotationZGate::label = "RZ";

Pennylane::RotationZGate Pennylane::RotationZGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::RotationZGate::label, parameters, 1);
    return Pennylane::RotationZGate(parameters[0], wires);
}

Pennylane::RotationZGate::RotationZGate(double rotationAngle, const vector<unsigned int>& wires)
    : first(std::pow(M_E, CplxType(0, -rotationAngle / 2)))
    , second(std::pow(M_E, CplxType(0, rotationAngle / 2)))
    , matrix{
      first, 0,
      0, second }
    , allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

void Pennylane::RotationZGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType shift1 = first;
    CplxType shift2 = second;

    if (inverse == true) {
        shift1 = conj(first);
        shift2 = conj(second);
    }

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[0]] *= shift1;
        shiftedState[indices[1]] *= shift2;
    }
}

const double Pennylane::RotationZGate::generatorScalingFactor{ -0.5 };

void Pennylane::RotationZGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    unique_ptr<Pennylane::ZGate> gate;
    gate->applyKernel(state, indices, externalIndices, false);
}


// -------------------------------------------------------------------------------------------------------------

const string Pennylane::PhaseShiftGate::label = "PhaseShift";

Pennylane::PhaseShiftGate Pennylane::PhaseShiftGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::PhaseShiftGate::label, parameters, 1);
    return Pennylane::PhaseShiftGate(parameters[0], wires);
}

Pennylane::PhaseShiftGate::PhaseShiftGate(double rotationAngle, const vector<unsigned int>& wires)
    : shift(std::pow(M_E, CplxType(0, rotationAngle)))
    , matrix{
      1, 0,
      0, shift }
    , allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

void Pennylane::PhaseShiftGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType s = (inverse == true) ? conj(shift) : shift;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[1]] *= s;
    }
}

const double Pennylane::PhaseShiftGate::generatorScalingFactor{ 1.0 };

void Pennylane::PhaseShiftGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[0]] = 0;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::GeneralRotationGate::label = "Rot";

Pennylane::GeneralRotationGate Pennylane::GeneralRotationGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::GeneralRotationGate::label, parameters, 3);
    return Pennylane::GeneralRotationGate(parameters[0], parameters[1], parameters[2], wires);
}

Pennylane::GeneralRotationGate::GeneralRotationGate(double phi, double theta, double omega, const vector<unsigned int>& wires)
    : c(std::cos(theta / 2), 0)
    , s(std::sin(theta / 2), 0)
    , r1(c* std::pow(M_E, CplxType(0, (-phi - omega) / 2)))
    , r2(-s * std::pow(M_E, CplxType(0, (phi - omega) / 2)))
    , r3(s* std::pow(M_E, CplxType(0, (-phi + omega) / 2)))
    , r4(c* std::pow(M_E, CplxType(0, (phi + omega) / 2)))
    , matrix{
      r1, r2,
      r3, r4 }
    , allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

void Pennylane::GeneralRotationGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType t1 = r1;
    CplxType t2 = r2;
    CplxType t3 = r3;
    CplxType t4 = r4;

    if (inverse == true) {
        t1 = conj(r1);
        t2 *= -1;
        t3 *= -1;
        t4 = conj(t4);
    }

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        CplxType v1 = shiftedState[indices[1]];
        shiftedState[indices[0]] = t1 * v0 + t2 * v1;
        shiftedState[indices[1]] = t3 * v0 + t4 * v1;
    }
}

// -------------------------------------------------------------------------------------------------------------

Pennylane::TwoQubitGate::TwoQubitGate()
    : AbstractGate(2)
{}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CNOTGate::label = "CNOT";

Pennylane::CNOTGate Pennylane::CNOTGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CNOTGate::label, parameters, 0);
    return Pennylane::CNOTGate(wires);
}

const std::vector<CplxType> Pennylane::CNOTGate::matrix{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0 };

const std::vector<CplxType> Pennylane::CNOTGate::targetMatrix = Pennylane::XGate::matrix;

Pennylane::CNOTGate::CNOTGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CNOTGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        swap(shiftedState[indices[2]], shiftedState[indices[3]]);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::SWAPGate::label = "SWAP";

Pennylane::SWAPGate Pennylane::SWAPGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::SWAPGate::label, parameters, 0);
    return Pennylane::SWAPGate(wires);
}

const std::vector<CplxType> Pennylane::SWAPGate::matrix{
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1 };

Pennylane::SWAPGate::SWAPGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({})
    , targetWires(wires)
{}

void Pennylane::SWAPGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        swap(shiftedState[indices[1]], shiftedState[indices[2]]);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CZGate::label = "CZ";

Pennylane::CZGate Pennylane::CZGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CZGate::label, parameters, 0);
    return Pennylane::CZGate(wires);
}

const std::vector<CplxType> Pennylane::CZGate::matrix{
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, -1 };

const std::vector<CplxType> Pennylane::CZGate::targetMatrix = Pennylane::ZGate::matrix;

Pennylane::CZGate::CZGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CZGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[3]] *= -1;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CRotationXGate::label = "CRX";

Pennylane::CRotationXGate Pennylane::CRotationXGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CRotationXGate::label, parameters, 1);
    return Pennylane::CRotationXGate(parameters[0], wires);
}

Pennylane::CRotationXGate::CRotationXGate(double rotationAngle, const vector<unsigned int>& wires)
    : c(std::cos(rotationAngle / 2), 0)
    , js(0, std::sin(-rotationAngle / 2))
    , matrix{
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, c, js,
      0, 0, js, c }
    , targetMatrix{
      c, js,
      js, c }
    , allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CRotationXGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType js_ = (inverse == true) ? -js : js;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[2]];
        CplxType v1 = shiftedState[indices[3]];
        shiftedState[indices[2]] = c * v0 + js_ * v1;
        shiftedState[indices[3]] = js_ * v0 + c * v1;
    }
}

const double Pennylane::CRotationXGate::generatorScalingFactor{ -0.5 };

void Pennylane::CRotationXGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
        swap(shiftedState[indices[2]], shiftedState[indices[3]]);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CRotationYGate::label = "CRY";

Pennylane::CRotationYGate Pennylane::CRotationYGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CRotationYGate::label, parameters, 1);
    return Pennylane::CRotationYGate(parameters[0], wires);
}

Pennylane::CRotationYGate::CRotationYGate(double rotationAngle, const vector<unsigned int>& wires)
    : c(std::cos(rotationAngle / 2), 0)
    , s(std::sin(rotationAngle / 2), 0)
    , matrix{
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, c, -s,
      0, 0, s, c }
    , targetMatrix{
      c, -s,
      s, c }
    , allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CRotationYGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType s_ = (inverse == true) ? -s : s;

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[2]];
        CplxType v1 = shiftedState[indices[3]];
        shiftedState[indices[2]] = c * v0 - s_ * v1;
        shiftedState[indices[3]] = s_ * v0 + c * v1;
    }
}

const double Pennylane::CRotationYGate::generatorScalingFactor{ -0.5 };

void Pennylane::CRotationYGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[0]];
        shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
        shiftedState[indices[2]] = -IMAG * shiftedState[indices[3]];
        shiftedState[indices[3]] = IMAG * v0;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CRotationZGate::label = "CRZ";

Pennylane::CRotationZGate Pennylane::CRotationZGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CRotationZGate::label, parameters, 1);
    return Pennylane::CRotationZGate(parameters[0], wires);
}

Pennylane::CRotationZGate::CRotationZGate(double rotationAngle, const vector<unsigned int>& wires)
    : first(std::pow(M_E, CplxType(0, -rotationAngle / 2)))
    , second(std::pow(M_E, CplxType(0, rotationAngle / 2)))
    , matrix{
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, first, 0,
      0, 0, 0, second }
    , targetMatrix{
      first, 0,
      0, second }
    , allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CRotationZGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType shift1 = first;
    CplxType shift2 = second;

    if (inverse == true) {
        shift1 = conj(first);
        shift2 = conj(second);
    }

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[2]] *= shift1;
        shiftedState[indices[3]] *= shift2;
    }
}

const double Pennylane::CRotationZGate::generatorScalingFactor{ -0.5 };

void Pennylane::CRotationZGate::applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices) {
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        shiftedState[indices[0]] = shiftedState[indices[1]] = 0;
        shiftedState[indices[3]] *= -1;
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CGeneralRotationGate::label = "CRot";

Pennylane::CGeneralRotationGate Pennylane::CGeneralRotationGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CGeneralRotationGate::label, parameters, 3);
    return Pennylane::CGeneralRotationGate(parameters[0], parameters[1], parameters[2], wires);
}

Pennylane::CGeneralRotationGate::CGeneralRotationGate(double phi, double theta, double omega, const vector<unsigned int>& wires)
    : c(std::cos(theta / 2), 0)
    , s(std::sin(theta / 2), 0)
    , r1(c* std::pow(M_E, CplxType(0, (-phi - omega) / 2)))
    , r2(-s * std::pow(M_E, CplxType(0, (phi - omega) / 2)))
    , r3(s* std::pow(M_E, CplxType(0, (-phi + omega) / 2)))
    , r4(c* std::pow(M_E, CplxType(0, (phi + omega) / 2)))
    , matrix{
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, r1, r2,
      0, 0, r3, r4 }
    , targetMatrix{
      r1, r2,
      r3, r4 }
    , allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1]})
{}

void Pennylane::CGeneralRotationGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse) {
    CplxType t1 = r1;
    CplxType t2 = r2;
    CplxType t3 = r3;
    CplxType t4 = r4;

    if (inverse == true) {
        t1 = conj(r1);
        t2 *= -1;
        t3 *= -1;
        t4 = conj(t4);
    }

    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        CplxType v0 = shiftedState[indices[2]];
        CplxType v1 = shiftedState[indices[3]];
        shiftedState[indices[2]] = t1 * v0 + t2 * v1;
        shiftedState[indices[3]] = t3 * v0 + t4 * v1;
    }
}

// -------------------------------------------------------------------------------------------------------------

Pennylane::ThreeQubitGate::ThreeQubitGate()
    : AbstractGate(3)
{}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::ToffoliGate::label = "Toffoli";

Pennylane::ToffoliGate Pennylane::ToffoliGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::ToffoliGate::label, parameters, 0);
    return Pennylane::ToffoliGate(wires);
}

const std::vector<CplxType> Pennylane::ToffoliGate::matrix{
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0 };

Pennylane::ToffoliGate::ToffoliGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({wires[0], wires[1]})
    , targetWires(wires[2])
{}

void Pennylane::ToffoliGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        swap(shiftedState[indices[6]], shiftedState[indices[7]]);
    }
}

// -------------------------------------------------------------------------------------------------------------

const string Pennylane::CSWAPGate::label = "CSWAP";

Pennylane::CSWAPGate Pennylane::CSWAPGate::create(const vector<double>& parameters, const vector<unsigned int>& wires) {
    validateLength(Pennylane::CSWAPGate::label, parameters, 0);
    return Pennylane::CSWAPGate(wires);
}

const std::vector<CplxType> Pennylane::CSWAPGate::matrix{
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1 };

const std::vector<CplxType> Pennylane::CSWAPGate::targetMatrix = Pennylane::SWAPGate::matrix;

Pennylane::CSWAPGate::CSWAPGate(const vector<unsigned int>& wires)
    : allWires(wires),
      controlWires({wires[0]})
    , targetWires({wires[1], wires[2]})
{}

void Pennylane::CSWAPGate::applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool) {
    // gate is its own inverse
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedState = state.arr + externalIndex;
        swap(shiftedState[indices[5]], shiftedState[indices[6]]);
    }
}

// -------------------------------------------------------------------------------------------------------------
// General gates

const string Pennylane::QubitUnitary::label = "QubitUnitary";

Pennylane::QubitUnitary::QubitUnitary(const int numQubits, std::vector<CplxType> const &mx, const vector<unsigned int>& controlWires, const vector<unsigned int>& targetWires)
 : AbstractGate(numQubits),
   matrix(mx),
   controlWires(controlWires),
   targetWires(targetWires),
   allWires(controlWires) {allWires.insert(allWires.end(), this->targetWires.begin(), this->targetWires.end());} ;



// -------------------------------------------------------------------------------------------------------------

template<class GateType>
static void addToDispatchTable(map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&, const vector<unsigned int>&)>>& dispatchTable) {
    dispatchTable.emplace(GateType::label, [](const vector<double>& parameters, const vector<unsigned int>& wires) { return make_unique<GateType>(GateType::create(parameters, wires)); });
}

static map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&, const vector<unsigned int>&)>> createDispatchTable() {
    map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&, const vector<unsigned int>&)>> dispatchTable;
    addToDispatchTable<Pennylane::XGate>(dispatchTable);
    addToDispatchTable<Pennylane::YGate>(dispatchTable);
    addToDispatchTable<Pennylane::ZGate>(dispatchTable);
    addToDispatchTable<Pennylane::HadamardGate>(dispatchTable);
    addToDispatchTable<Pennylane::SGate>(dispatchTable);
    addToDispatchTable<Pennylane::TGate>(dispatchTable);
    addToDispatchTable<Pennylane::RotationXGate>(dispatchTable);
    addToDispatchTable<Pennylane::RotationYGate>(dispatchTable);
    addToDispatchTable<Pennylane::RotationZGate>(dispatchTable);
    addToDispatchTable<Pennylane::PhaseShiftGate>(dispatchTable);
    addToDispatchTable<Pennylane::GeneralRotationGate>(dispatchTable);
    addToDispatchTable<Pennylane::CNOTGate>(dispatchTable);
    addToDispatchTable<Pennylane::SWAPGate>(dispatchTable);
    addToDispatchTable<Pennylane::CZGate>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationXGate>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationYGate>(dispatchTable);
    addToDispatchTable<Pennylane::CRotationZGate>(dispatchTable);
    addToDispatchTable<Pennylane::CGeneralRotationGate>(dispatchTable);
    addToDispatchTable<Pennylane::ToffoliGate>(dispatchTable);
    addToDispatchTable<Pennylane::CSWAPGate>(dispatchTable);
    return dispatchTable;
}

static const map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&, const vector<unsigned int>&)>> dispatchTable = createDispatchTable();

unique_ptr<Pennylane::AbstractGate> Pennylane::constructGate(const string& label, const vector<double>& parameters, const vector<unsigned int>& wires) {
    auto dispatchTableIterator = dispatchTable.find(label);
    if (dispatchTableIterator == dispatchTable.end())
        throw std::invalid_argument(label + " is not a supported gate type");

    std::cout << "Wires for CNOT: ";
    for (auto it : wires){
        std::cout << it << " ";
    }

    return dispatchTableIterator->second(parameters, wires);
}

unique_ptr<Pennylane::AbstractGate> Pennylane::constructGate(const vector<CplxType>& matrix, const vector<unsigned int>& controlWires, const vector<unsigned int>& targetWires) {
    // TODO: validate
    auto num_qubits = int(log2(matrix.size())/2);
    return make_unique<QubitUnitary>(QubitUnitary(num_qubits, matrix, controlWires, targetWires));
}
