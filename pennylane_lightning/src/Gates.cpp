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

#include "Gates.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>

#include "Util.hpp"

using std::conj;
using std::function;
using std::map;
using std::string;
using std::swap;
using std::unique_ptr;
using std::vector;

template <class Derived>
Pennylane::AbstractGate<Derived>::AbstractGate(size_t numQubits)
    : numQubits(numQubits), length(exp2(numQubits)) {}

template <class Derived>
void Pennylane::AbstractGate<Derived>::applyKernel(
    const StateVector<decltype(Derived::precision_)> &state,
    const std::vector<size_t> &indices,
    const std::vector<size_t> &externalIndices, bool inverse) {
    const vector<CplxType> &matrix = asMatrix();
    assert(indices.size() == length);

    vector<CplxType> v(indices.size());
    for (const size_t &externalIndex : externalIndices) {
        CplxType *shiftedState = state.arr + externalIndex;
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
                    shiftedState[index] += conj(matrix[baseIndex + i]) * v[j];
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

const double Pennylane::AbstractGate::generatorScalingFactor{};
void Pennylane::AbstractGate::applyGenerator(const StateVector &,
                                             const std::vector<size_t> &,
                                             const std::vector<size_t> &) {
    throw NotImplementedException();
}

template class Pennylane::XGate<float>;
template class Pennylane::XGate<double>;

template class Pennylane::YGate<float>;
template class Pennylane::YGate<double>;

template class Pennylane::ZGate<float>;
template class Pennylane::ZGate<double>;

template class Pennylane::HadamardGate<float>;
template class Pennylane::HadamardGate<double>;

template class Pennylane::SGate<float>;
template class Pennylane::SGate<double>;

template class Pennylane::TGate<float>;
template class Pennylane::TGate<double>;

template class Pennylane::RotationXGate<float>;
template class Pennylane::RotationXGate<double>;

template class Pennylane::RotationYGate<float>;
template class Pennylane::RotationYGate<double>;

template class Pennylane::RotationZGate<float>;
template class Pennylane::RotationZGate<double>;

template class Pennylane::PhaseShiftGate<float>;
template class Pennylane::PhaseShiftGate<double>;

template class Pennylane::CNOTGate<float>;
template class Pennylane::CNOTGate<double>;

template class Pennylane::SWAPGate<float>;
template class Pennylane::SWAPGate<double>;

template class Pennylane::CZGate<float>;
template class Pennylane::CZGate<double>;

template class Pennylane::CRotationXGate<float>;
template class Pennylane::CRotationXGate<double>;

template class Pennylane::CRotationYGate<float>;
template class Pennylane::CRotationYGate<double>;

template class Pennylane::CRotationZGate<float>;
template class Pennylane::CRotationZGate<double>;

template class Pennylane::CGeneralRotationGate<float>;
template class Pennylane::CGeneralRotationGate<double>;

template class Pennylane::ToffoliGate<float>;
template class Pennylane::ToffoliGate<double>;

template class Pennylane::CSWAPGate<float>;
template class Pennylane::CSWAPGate<double>;

// Pennylane::TwoQubitGate::TwoQubitGate() : AbstractGate(2) {}

// -------------------------------------------------------------------------------------------------------------

template <class GateType>
static void addToDispatchTable(
    map<string,
        function<unique_ptr<Pennylane::AbstractGate>(const vector<double> &)>>
        &dispatchTable) {
    dispatchTable.emplace(
        GateType::label, [](const vector<double> &parameters) {
            return make_unique<GateType>(GateType::create(parameters));
        });
}

static map<string, function<unique_ptr<Pennylane::AbstractGate>(
                       const vector<double> &)>>
createDispatchTable() {
    map<string,
        function<unique_ptr<Pennylane::AbstractGate>(const vector<double> &)>>
        dispatchTable;
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

static const map<string, function<unique_ptr<Pennylane::AbstractGate>(
                             const vector<double> &)>>
    dispatchTable = createDispatchTable();

unique_ptr<Pennylane::AbstractGate>
Pennylane::constructGate(const string &label,
                         const vector<double> &parameters) {
    auto dispatchTableIterator = dispatchTable.find(label);
    if (dispatchTableIterator == dispatchTable.end())
        throw std::invalid_argument(label + " is not a supported gate type");

    return dispatchTableIterator->second(parameters);
}
