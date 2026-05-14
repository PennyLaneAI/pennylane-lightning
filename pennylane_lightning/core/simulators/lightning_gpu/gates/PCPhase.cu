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
#include <cstddef>

#include "cuError.hpp"
#include "cuda_helpers.hpp"

namespace Pennylane::LightningGPU {
namespace {

// NOTE: structs can be directly passed to kernels, avoids having to H2D memcopy
constexpr std::size_t maxPCPhaseWires = 64;
struct WireInfo {
    std::uint64_t ctrl_mask = 0;
    std::uint64_t ctrl_vals = 0;
    std::size_t num_tgts = 0;
    int tgt_wires[maxPCPhaseWires]{};
    WireInfo(const std::vector<int> &ctrl_wires,
             const std::vector<int> &ctrl_vals_vec,
             const std::vector<int> &tgt_wires_vec) {
        num_tgts = tgt_wires_vec.size();
        std::copy(tgt_wires_vec.begin(), tgt_wires_vec.end(), tgt_wires);
        std::tie(ctrl_mask, ctrl_vals) =
            makeCtrlMaskValues(ctrl_wires, ctrl_vals_vec);
    }
    auto makeCtrlMaskValues(const std::vector<int> &wires,
                            const std::vector<int> &values)
        -> std::pair<std::uint64_t, std::uint64_t> {
        std::uint64_t mask = 0;
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < wires.size(); ++i) {
            std::uint64_t bit = 1ULL << static_cast<unsigned>(wires[i]);
            mask |= bit;
            if (values[i]) {
                value |= bit;
            }
        }
        return {mask, value};
    }
};

template <class GPUDataT>
__global__ void applyPCPhaseKernel(GPUDataT *sv, std::size_t sv_length,
                                   WireInfo params, std::size_t dimension,
                                   GPUDataT factor) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= sv_length) {
        return;
    }

    // Check if all control bits are set
    if ((index & params.ctrl_mask) != params.ctrl_vals) {
        return;
    }

    // Extract target bits from index
    std::size_t target_index = 0;
    for (std::size_t i = 0; i < params.num_tgts; i++) {
        // Check if target bit is set
        const std::size_t bit =
            (index >> static_cast<std::size_t>(params.tgt_wires[i])) & 1U;
        // Shift target index left and add bit
        target_index <<= 1U;
        // Add bit to target index
        target_index |= bit;
    }
    using RealT = decltype(factor.x);
    RealT phase_sign = (target_index < dimension) ? RealT{1.0} : RealT{-1.0};
    GPUDataT mult_factor{factor.x, phase_sign * factor.y};
    sv[index] = Util::Cmul(sv[index], mult_factor);
}

template <class GPUDataT>
__global__ void applyDiagKernel(GPUDataT *sv, std::size_t sv_length,
                                WireInfo info, const GPUDataT *diag) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= sv_length) {
        return;
    }

    // Check if all control bits are set
    if ((index & info.ctrl_mask) != info.ctrl_vals) {
        return;
    }

    // Extract target bits from index and use as index into diagonal array
    std::size_t target_index = 0;
    for (std::size_t i = 0; i < info.num_tgts; i++) {
        // Check if target bit is set
        const std::size_t bit =
            (index >> static_cast<std::size_t>(info.tgt_wires[i])) & 1U;
        // Shift target index left and add bit
        target_index <<= 1U;
        // Add bit to target index
        target_index |= bit;
    }

    sv[index] = Util::Cmul(sv[index], diag[target_index]);
}

} // namespace

template <class GPUDataT, class PrecisionT>
void applyPCPhase_CUDA(GPUDataT *sv, std::size_t sv_length,
                       const std::vector<int> &ctrl_wires,
                       const std::vector<int> &ctrl_values,
                       const std::vector<int> &tgts_wires,
                       std::size_t dimension, PrecisionT phase, int device_id,
                       cudaStream_t stream_id) {
    PL_CUDA_IS_SUCCESS(cudaSetDevice(device_id));

    const auto factor = Util::complexToCu(
        std::complex<PrecisionT>{std::cos(phase), std::sin(phase)});

    WireInfo wire_info(ctrl_wires, ctrl_values, tgts_wires);

    const std::size_t threads_per_block = 256;
    const std::size_t blocks_per_grid = std::max<std::size_t>(
        (sv_length + threads_per_block - 1) / threads_per_block, 1);
    applyPCPhaseKernel<GPUDataT>
        <<<blocks_per_grid, threads_per_block, 0, stream_id>>>(
            sv, sv_length, wire_info, dimension, factor);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
    PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(stream_id));
}

template <class GPUDataT>
void applyDiag_CUDA(GPUDataT *sv, std::size_t sv_length,
                    const std::vector<int> &ctrl_wires,
                    const std::vector<int> &ctrl_vals,
                    const std::vector<int> &tgts_wires, const GPUDataT *diag,
                    int device_id, cudaStream_t stream_id) {
    PL_CUDA_IS_SUCCESS(cudaSetDevice(device_id));

    const WireInfo wire_info(ctrl_wires, ctrl_vals, tgts_wires);
    const std::size_t threads_per_block = 256;
    const std::size_t blocks_per_grid = std::max<std::size_t>(
        (sv_length + threads_per_block - 1) / threads_per_block, 1);
    applyDiagKernel<GPUDataT>
        <<<blocks_per_grid, threads_per_block, 0, stream_id>>>(sv, sv_length,
                                                               wire_info, diag);
    PL_CUDA_IS_SUCCESS(cudaGetLastError());
    PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(stream_id));
}

// Explicit template instantiations
template void applyPCPhase_CUDA<cuComplex, float>(cuComplex *, std::size_t,
                                                  const std::vector<int> &,
                                                  const std::vector<int> &,
                                                  const std::vector<int> &,
                                                  std::size_t, float, int,
                                                  cudaStream_t);
template void applyPCPhase_CUDA<cuDoubleComplex, double>(
    cuDoubleComplex *, std::size_t, const std::vector<int> &,
    const std::vector<int> &, const std::vector<int> &, std::size_t, double,
    int, cudaStream_t);

template void applyDiag_CUDA<cuComplex>(cuComplex *, std::size_t,
                                        const std::vector<int> &,
                                        const std::vector<int> &,
                                        const std::vector<int> &,
                                        const cuComplex *, int, cudaStream_t);
template void applyDiag_CUDA<cuDoubleComplex>(cuDoubleComplex *, std::size_t,
                                              const std::vector<int> &,
                                              const std::vector<int> &,
                                              const std::vector<int> &,
                                              const cuDoubleComplex *, int,
                                              cudaStream_t);

} // namespace Pennylane::LightningGPU
