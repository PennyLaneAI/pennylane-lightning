// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "AssignKernelMap_Default.hpp"
#include "GateOperation.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"

using namespace Pennylane;
using namespace Pennylane::KernelMap;

using Gates::GateOperation;
using Gates::GeneratorOperation;
using Gates::KernelType;
using Gates::MatrixOperation;
using Util::full_domain;
using Util::in_between_closed;
using Util::larger_than;
using Util::larger_than_equal_to;
using Util::less_than;
using Util::less_than_equal_to;

namespace Pennylane::KernelMap::Internal {
constexpr static auto all_qubit_numbers = Util::full_domain<size_t>();

void assignKernelsForGateOp_Default() {
    auto &instance = OperationKernelMap<GateOperation>::getInstance();

    instance.assignKernelForOp(GateOperation::Identity, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::PauliZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::Hadamard, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::S, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::T, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::PhaseShift, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::Rot, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    /* Two-qubit gates */
    instance.assignKernelForOp(GateOperation::CNOT, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::ControlledPhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::SWAP, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingXX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingXY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingYY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::IsingZZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::CRot, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);

    /* Three-qubit gates */
    instance.assignKernelForOp(GateOperation::Toffoli, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::PI);
    instance.assignKernelForOp(GateOperation::CSWAP, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::PI);

    /* QChem gates */
    instance.assignKernelForOp(GateOperation::SingleExcitation, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::SingleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::SingleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GateOperation::DoubleExcitation, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::PI);
    instance.assignKernelForOp(GateOperation::DoubleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::PI);
    instance.assignKernelForOp(GateOperation::DoubleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::PI);

    /* Multi-qubit gates */
    instance.assignKernelForOp(GateOperation::MultiRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               Gates::KernelType::LM);
}

void assignKernelsForGeneratorOp_Default() {
    auto &instance = OperationKernelMap<GeneratorOperation>::getInstance();

    instance.assignKernelForOp(GeneratorOperation::PhaseShift, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::RZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingXX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingXY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingYY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::IsingZZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRX, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRY, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::CRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::ControlledPhaseShift,
                               all_threading, all_memory_model,
                               all_qubit_numbers, KernelType::LM);

    instance.assignKernelForOp(GeneratorOperation::SingleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::SingleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::SingleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::LM);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitation,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::PI);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitationPlus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::PI);
    instance.assignKernelForOp(GeneratorOperation::DoubleExcitationMinus,
                               all_threading, all_memory_model,
                               all_qubit_numbers, Gates::KernelType::PI);

    instance.assignKernelForOp(GeneratorOperation::MultiRZ, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
}
void assignKernelsForMatrixOp_Default() {
    auto &instance = OperationKernelMap<MatrixOperation>::getInstance();

    instance.assignKernelForOp(MatrixOperation::SingleQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(MatrixOperation::TwoQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::LM);
    instance.assignKernelForOp(MatrixOperation::MultiQubitOp, all_threading,
                               all_memory_model, all_qubit_numbers,
                               KernelType::PI);
}
} // namespace Pennylane::KernelMap::Internal
