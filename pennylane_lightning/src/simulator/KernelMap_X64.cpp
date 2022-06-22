
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
 * Assign kernel map
 */

#include "KernelMap.hpp"
#include "AssignKernelMap_Default.hpp"
#include "AssignKernelMap_AVX2.hpp"
#include "AssignKernelMap_AVX512.hpp"
#include "RuntimeInfo.hpp"

namespace Pennylane::KernelMap::Internal {

int assignKernelsForGateOp() {
    assignKernelsForGateOp_Default();

    if(RuntimeInfo::AVX2()) {
        assignKernelsForGateOp_AVX2(CPUMemoryModel::Aligned256);
        if(!RuntimeInfo::AVX512()) {
            assignKernelsForGateOp_AVX2(CPUMemoryModel::Aligned512);
        }
    }
    if(RuntimeInfo::AVX512()) {
        assignKernelsForGateOp_AVX512(CPUMemoryModel::Aligned512);
    }
    return 1;
}
int assignKernelsForGeneratorOp() {
    assignKernelsForGeneratorOp_Default();

    if(RuntimeInfo::AVX2()) {
        assignKernelsForGeneratorOp_AVX2(CPUMemoryModel::Aligned256);
        if(!RuntimeInfo::AVX512()) {
            assignKernelsForGeneratorOp_AVX2(CPUMemoryModel::Aligned512);
        }
    }
    if(RuntimeInfo::AVX512()) {
        assignKernelsForGeneratorOp_AVX512(CPUMemoryModel::Aligned512);
    }
    return 1;
}
int assignKernelsForMatrixOp() {
    assignKernelsForMatrixOp_Default();

    if(RuntimeInfo::AVX2()) {
        assignKernelsForMatrixOp_AVX2(CPUMemoryModel::Aligned256);
        if(!RuntimeInfo::AVX512()) {
            assignKernelsForMatrixOp_AVX2(CPUMemoryModel::Aligned512);
        }
    }
    if(RuntimeInfo::AVX512()) {
        assignKernelsForMatrixOp_AVX512(CPUMemoryModel::Aligned512);
    }
    return 1;
}
} // namespace Pennylane::KernelMap::Internal
