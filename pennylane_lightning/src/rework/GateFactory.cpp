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

#include <functional>
#include <map>

#include "GateFactory.hpp"

using std::function;
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

template<class GateType>
static void addToDispatchTable(map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&)>>& dispatchTable) {
    dispatchTable.emplace(GateType::label, [](const vector<double>& parameters) { return std::make_unique<GateType>(GateType::create(parameters)); });
}

static map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&)>> createDispatchTable() {
    map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&)>> dispatchTable;
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

static const map<string, function<unique_ptr<Pennylane::AbstractGate>(const vector<double>&)>> dispatchTable = createDispatchTable();

unique_ptr<Pennylane::AbstractGate> Pennylane::constructGate(const string& label, const vector<double>& parameters) {
    auto dispatchTableIterator = dispatchTable.find(label);
    if (dispatchTableIterator == dispatchTable.end())
        throw std::invalid_argument(label + " is not a supported gate type");

    return dispatchTableIterator->second(parameters);
}
