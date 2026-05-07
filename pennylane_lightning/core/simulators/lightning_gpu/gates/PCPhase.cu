// Copyright 2022-2023 Xanadu Quantum Technologies Inc.
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
 * @file PCPhase.cu
 */
#include "PCPhase.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "Error.hpp"
#include "cuError.hpp"
#include "cuda_helpers.hpp"

namespace Pennylane::LightningGPU {
namespace {
constexpr std::size_t maxPCPhaseWires = 64;

struct PCPhaseWireList {
    std::size_t size = 0;
    int wires[maxPCPhaseWires]{};
    int values[maxPCPhaseWires]{};
};

template <class GPUDataT, class PrecisionT>
auto makeCudaComplex(PrecisionT real, PrecisionT imag) -> GPUDataT {
    if constexpr (std::is_same_v<GPUDataT, cuComplex>) {
        return make_cuFloatComplex(static_cast<float>(real),
                                   static_cast<float>(imag));
    } else {
        return make_cuDoubleComplex(static_cast<double>(real),
                                    static_cast<double>(imag));
    }
}

inline auto makePCPhaseWireList(const int *wires, std::size_t num_wires,
                                const int *values = nullptr)
    -> PCPhaseWireList {
    PL_ABORT_IF(num_wires > maxPCPhaseWires,
                "PCPhase supports at most 64 wires.");

    PCPhaseWireList wire_list{};
    wire_list.size = num_wires;
    for (std::size_t i = 0; i < num_wires; i++) {
        wire_list.wires[i] = wires[i];
        if (values != nullptr) {
            wire_list.values[i] = values[i];
        }
    }
    return wire_list;
}

template <class GPUDataT>
__global__ void applyPCPhaseKernel(GPUDataT *sv, std::size_t sv_length,
                                   PCPhaseWireList tgts,
                                   std::size_t dimension, GPUDataT upper,
                                   GPUDataT lower) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= sv_length) {
        return;
    }

    std::size_t target_index = 0;
    for (std::size_t i = 0; i < tgts.size; i++) {
        target_index =
            (target_index << 1U) |
            ((index >> static_cast<std::size_t>(tgts.wires[i])) & 1U);
    }

    sv[index] =
        Util::Cmul(sv[index], target_index < dimension ? upper : lower);
}

template <class GPUDataT>
__global__ void applyControlledPCPhaseKernel(
    GPUDataT *sv, std::size_t sv_length, PCPhaseWireList ctrls,
    PCPhaseWireList tgts, std::size_t dimension, GPUDataT upper,
    GPUDataT lower) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= sv_length) {
        return;
    }

    for (std::size_t i = 0; i < ctrls.size; i++) {
        const int bit =
            static_cast<int>((index >> static_cast<std::size_t>(ctrls.wires[i])) &
                             1U);
        if (bit != ctrls.values[i]) {
            return;
        }
    }

    std::size_t target_index = 0;
    for (std::size_t i = 0; i < tgts.size; i++) {
        target_index =
            (target_index << 1U) |
            ((index >> static_cast<std::size_t>(tgts.wires[i])) & 1U);
    }

    sv[index] =
        Util::Cmul(sv[index], target_index < dimension ? upper : lower);
}

template <class GPUDataT, class PrecisionT>
void applyPCPhase_CUDA_call(GPUDataT *sv, std::size_t sv_length,
                            const int *tgts, std::size_t num_tgts,
                            std::size_t dimension, PrecisionT phase,
                            std::size_t thread_per_block, int device_id,
                            cudaStream_t stream_id) {
    PL_CUDA_IS_SUCCESS(cudaSetDevice(device_id));

    const auto target_list = makePCPhaseWireList(tgts, num_tgts);
    const auto upper = makeCudaComplex<GPUDataT>(std::cos(phase),
                                                std::sin(phase));
    const auto lower = makeCudaComplex<GPUDataT>(std::cos(phase),
                                                -std::sin(phase));

    const std::size_t block_per_grid =
        std::max<std::size_t>((sv_length + thread_per_block - 1) /
                                  thread_per_block,
                              1);
    applyPCPhaseKernel<GPUDataT>
        <<<block_per_grid, thread_per_block, 0, stream_id>>>(
            sv, sv_length, target_list, dimension, upper, lower);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}

template <class GPUDataT, class PrecisionT>
void applyControlledPCPhase_CUDA_call(
    GPUDataT *sv, std::size_t sv_length, const int *ctrls,
    const int *ctrl_values, std::size_t num_ctrls, const int *tgts,
    std::size_t num_tgts, std::size_t dimension, PrecisionT phase,
    std::size_t thread_per_block, int device_id, cudaStream_t stream_id) {
    PL_CUDA_IS_SUCCESS(cudaSetDevice(device_id));

    const auto control_list = makePCPhaseWireList(ctrls, num_ctrls,
                                                  ctrl_values);
    const auto target_list = makePCPhaseWireList(tgts, num_tgts);
    const auto upper = makeCudaComplex<GPUDataT>(std::cos(phase),
                                                std::sin(phase));
    const auto lower = makeCudaComplex<GPUDataT>(std::cos(phase),
                                                -std::sin(phase));

    const std::size_t block_per_grid =
        std::max<std::size_t>((sv_length + thread_per_block - 1) /
                                  thread_per_block,
                              1);
    applyControlledPCPhaseKernel<GPUDataT>
        <<<block_per_grid, thread_per_block, 0, stream_id>>>(
            sv, sv_length, control_list, target_list, dimension, upper, lower);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
}
} // namespace

void applyPCPhase_CUDA(cuComplex *sv, std::size_t sv_length, const int *tgts,
                       std::size_t num_tgts, std::size_t dimension,
                       float phase, std::size_t thread_per_block,
                       int device_id, cudaStream_t stream_id) {
    applyPCPhase_CUDA_call(sv, sv_length, tgts, num_tgts, dimension, phase,
                           thread_per_block, device_id, stream_id);
}

void applyPCPhase_CUDA(cuDoubleComplex *sv, std::size_t sv_length,
                       const int *tgts, std::size_t num_tgts,
                       std::size_t dimension, double phase,
                       std::size_t thread_per_block, int device_id,
                       cudaStream_t stream_id) {
    applyPCPhase_CUDA_call(sv, sv_length, tgts, num_tgts, dimension, phase,
                           thread_per_block, device_id, stream_id);
}

void applyControlledPCPhase_CUDA(cuComplex *sv, std::size_t sv_length,
                                 const int *ctrls, const int *ctrl_values,
                                 std::size_t num_ctrls, const int *tgts,
                                 std::size_t num_tgts, std::size_t dimension,
                                 float phase, std::size_t thread_per_block,
                                 int device_id, cudaStream_t stream_id) {
    applyControlledPCPhase_CUDA_call(
        sv, sv_length, ctrls, ctrl_values, num_ctrls, tgts, num_tgts,
        dimension, phase, thread_per_block, device_id, stream_id);
}

void applyControlledPCPhase_CUDA(cuDoubleComplex *sv, std::size_t sv_length,
                                 const int *ctrls, const int *ctrl_values,
                                 std::size_t num_ctrls, const int *tgts,
                                 std::size_t num_tgts, std::size_t dimension,
                                 double phase, std::size_t thread_per_block,
                                 int device_id, cudaStream_t stream_id) {
    applyControlledPCPhase_CUDA_call(
        sv, sv_length, ctrls, ctrl_values, num_ctrls, tgts, num_tgts,
        dimension, phase, thread_per_block, device_id, stream_id);
}
} // namespace Pennylane::LightningGPU
