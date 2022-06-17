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
/**
 * @file DynamicDispatcher.cpp
 * Register all gate and generator implementations
 */
#include "DynamicDispatcher.hpp"
#include "RegisterKernel.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

namespace Pennylane::Internal {

__attribute__ ((target ("default")))
void registerKernelsFloat() {
    using namespace Pennylane::Gates;
    registerKernel<float, float, GateImplementationsLM>();
    registerKernel<float, float, GateImplementationsPI>();
}

__attribute__ ((target ("default")))
void registerKernelsDouble() {
    using namespace Pennylane::Gates;
    registerKernel<double, double, GateImplementationsLM>();
    registerKernel<double, double, GateImplementationsPI>();
}

template <class PrecisionT, class ParamT> int registerAllAvailableKernels() {
    if constexpr (std::is_same_v<PrecisionT, float> && std::is_same_v<ParamT, float>) {
        registerKernelsFloat();
        return 1;
    }
    if constexpr (std::is_same_v<PrecisionT, double> && std::is_same_v<ParamT, double>) {
        registerKernelsDouble();
        return 1;
    }

    return 0;
}

} // namespace Pennylane::Internal
