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
 * Defines kernel functions with AVX2
 */
#pragma once

// General implementations
#include "avx_common/AVX2Concept.hpp"

#include "avx_common/ApplyCNOT.hpp"
#include "avx_common/ApplyCZ.hpp"
#include "avx_common/ApplyHadamard.hpp"
#include "avx_common/ApplyIsingXX.hpp"
#include "avx_common/ApplyIsingYY.hpp"
#include "avx_common/ApplyIsingZZ.hpp"
#include "avx_common/ApplyPauliX.hpp"
#include "avx_common/ApplyPauliY.hpp"
#include "avx_common/ApplyPauliZ.hpp"
#include "avx_common/ApplyPhaseShift.hpp"
#include "avx_common/ApplyRX.hpp"
#include "avx_common/ApplyRY.hpp"
#include "avx_common/ApplyRZ.hpp"
#include "avx_common/ApplyS.hpp"
#include "avx_common/ApplySWAP.hpp"
#include "avx_common/ApplySingleQubitOp.hpp"
#include "avx_common/ApplyT.hpp"
#include "avx_common/TwoQubitGateHelper.hpp"

#include "Error.hpp"
#include "GateImplementationsAVXCommon.hpp"
#include "GateImplementationsLM.hpp"
#include "GateOperation.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "LinearAlgebra.hpp"
#include "Macros.hpp"

#include <immintrin.h>

#include <complex>
#include <vector>

namespace Pennylane::Gates {
class GateImplementationsAVX2
    : public GateImplementationsAVXCommon<GateImplementationsAVX2> {
  public:
    constexpr static KernelType kernel_id = KernelType::AVX2;
    constexpr static std::string_view name = "AVX2";
    constexpr static uint32_t packed_bytes = 32;

    constexpr static std::array implemented_matrices = {
        MatrixOperation::SingleQubitOp,
    };

    template <typename PrecisionT>
    static void
    applySingleQubitOp(std::complex<PrecisionT> *arr, const size_t num_qubits,
                       const std::complex<PrecisionT> *matrix,
                       const std::vector<size_t> &wires, bool inverse = false) {
        PL_ASSERT(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;

        using SingleQubitOpProdAVX2 =
            AVX::ApplySingleQubitOp<PrecisionT,
                                    packed_bytes / sizeof(PrecisionT)>;

        if (num_qubits <
            AVX::internal_wires_v<packed_bytes / sizeof(PrecisionT)>) {
            GateImplementationsLM::applySingleQubitOp(arr, num_qubits, matrix,
                                                      wires, inverse);
            return;
        }

        if constexpr (std::is_same_v<PrecisionT, float>) {
            switch (rev_wire) {
            case 0:
                SingleQubitOpProdAVX2::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
                return;
            case 1:
                SingleQubitOpProdAVX2::template applyInternal<1>(
                    arr, num_qubits, matrix, inverse);
                return;
            default:
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                     matrix, inverse);
                return;
            }
        } else if (std::is_same_v<PrecisionT, double>) {
            if (rev_wire == 0) {
                SingleQubitOpProdAVX2::template applyInternal<0>(
                    arr, num_qubits, matrix, inverse);
            } else {
                SingleQubitOpProdAVX2::applyExternal(arr, num_qubits, rev_wire,
                                                     matrix, inverse);
            }
        }
    }
};
} // namespace Pennylane::Gates
