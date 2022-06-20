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
 * @file
 * Register all gate and generator implementations for X86
 */
#include "DynamicDispatcher.hpp"
#include "RegisterKernel.hpp"
#include "RegisterKernels_x64.hpp"
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"
#include "RuntimeInfo.hpp"

namespace Pennylane::Internal {

template <class PrecisionT, class ParamT> int registerAllAvailableKernels() {
    using Pennylane::Util::RuntimeInfo;
    if constexpr (std::is_same_v<PrecisionT, float> && std::is_same_v<ParamT, float>) {
        registerKernel<float, float, Gates::GateImplementationsLM>();
        registerKernel<float, float, Gates::GateImplementationsPI>();

        if(RuntimeInfo::AVX2()) {
            registerKernelsAVX2_Float();
        }
        return 1;
    }
    if constexpr (std::is_same_v<PrecisionT, double> && std::is_same_v<ParamT, double>) {
        registerKernel<double, double, Gates::GateImplementationsLM>();
        registerKernel<double, double, Gates::GateImplementationsPI>();

        if(RuntimeInfo::AVX2()) {
            registerKernelsAVX2_Double();
        }
        return 1;
    }
    return 0;
}

} // namespace Pennylane::Internal
