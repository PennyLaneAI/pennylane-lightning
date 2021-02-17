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

#include "GateFactory.hpp"

using std::string;
using std::unique_ptr;
using std::vector;

// FIXME: This should be reworked to use a function dispatch table
unique_ptr<Pennylane::AbstractGate> Pennylane::constructGate(const string& label, const vector<double>& parameters) {
    std::unique_ptr<Pennylane::AbstractGate> gate;

    if (Pennylane::XGate::label == label) {
        gate = std::make_unique<Pennylane::XGate>(Pennylane::XGate::create(parameters));
    }
    else if (Pennylane::YGate::label == label) {
        gate = std::make_unique<Pennylane::YGate>(Pennylane::YGate::create(parameters));
    }
    else if (Pennylane::ZGate::label == label) {
        gate = std::make_unique<Pennylane::ZGate>(Pennylane::ZGate::create(parameters));
    }
    else if (Pennylane::HadamardGate::label == label) {
        gate = std::make_unique<Pennylane::HadamardGate>(Pennylane::HadamardGate::create(parameters));
    }
    else if (Pennylane::SGate::label == label) {
        gate = std::make_unique<Pennylane::SGate>(Pennylane::SGate::create(parameters));
    }
    else if (Pennylane::TGate::label == label) {
        gate = std::make_unique<Pennylane::TGate>(Pennylane::TGate::create(parameters));
    }
    else if (Pennylane::RotationXGate::label == label) {
        gate = std::make_unique<Pennylane::RotationXGate>(Pennylane::RotationXGate::create(parameters));
    }
    else if (Pennylane::RotationYGate::label == label) {
        gate = std::make_unique<Pennylane::RotationYGate>(Pennylane::RotationYGate::create(parameters));
    }
    else if (Pennylane::RotationZGate::label == label) {
        gate = std::make_unique<Pennylane::RotationZGate>(Pennylane::RotationZGate::create(parameters));
    }
    else if (Pennylane::PhaseShiftGate::label == label) {
        gate = std::make_unique<Pennylane::PhaseShiftGate>(Pennylane::PhaseShiftGate::create(parameters));
    }
    else if (Pennylane::GeneralRotationGate::label == label) {
        gate = std::make_unique<Pennylane::GeneralRotationGate>(Pennylane::GeneralRotationGate::create(parameters));
    }
    else if (Pennylane::CNOTGate::label == label) {
        gate = std::make_unique<Pennylane::CNOTGate>(Pennylane::CNOTGate::create(parameters));
    }
    else if (Pennylane::SWAPGate::label == label) {
        gate = std::make_unique<Pennylane::SWAPGate>(Pennylane::SWAPGate::create(parameters));
    }
    else if (Pennylane::CZGate::label == label) {
        gate = std::make_unique<Pennylane::CZGate>(Pennylane::CZGate::create(parameters));
    }
    else if (Pennylane::CRotationXGate::label == label) {
        gate = std::make_unique<Pennylane::CRotationXGate>(Pennylane::CRotationXGate::create(parameters));
    }
    else if (Pennylane::CRotationYGate::label == label) {
        gate = std::make_unique<Pennylane::CRotationYGate>(Pennylane::CRotationYGate::create(parameters));
    }
    else if (Pennylane::CRotationZGate::label == label) {
        gate = std::make_unique<Pennylane::CRotationZGate>(Pennylane::CRotationZGate::create(parameters));
    }
    else if (Pennylane::CGeneralRotationGate::label == label) {
        gate = std::make_unique<Pennylane::CGeneralRotationGate>(Pennylane::CGeneralRotationGate::create(parameters));
    }
    else if (Pennylane::ToffoliGate::label == label) {
        gate = std::make_unique<Pennylane::ToffoliGate>(Pennylane::ToffoliGate::create(parameters));
    }
    else if (Pennylane::CSWAPGate::label == label) {
        gate = std::make_unique<Pennylane::CSWAPGate>(Pennylane::CSWAPGate::create(parameters));
    }
    else {
        throw std::invalid_argument(label + " is not a valid gate type");
    }

    return gate;
}
