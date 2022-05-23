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
 * @file AvailableKernels.hpp
 * Defines available kernels. Be careful when including this file as
 * it also includes all gate implementations.
 */
#pragma once

#include "TypeList.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"
#include "cpu_kernels/QChemGateImplementations.hpp"

namespace Pennylane {
/**
 * @brief List of all available kernels (gate implementations).
 *
 * If you want to add another gate implementation, just add it to this type
 * list.
 * @rst
 * See :ref:`lightning_add_gate_implementation` for details.
 * @endrst
 */
using AvailableKernels = Util::TypeList<Gates::GateImplementationsLM,
                                        Gates::GateImplementationsPI, void>;
} // namespace Pennylane
