// Copyright 2024 Xanadu Quantum Technologies Inc.

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
 * @file tncuda_helpers.hpp
 */

#pragma once
#include <cutensornet.h>
#include <memory>
#include <utility>

#include "tncudaError.hpp"

namespace Pennylane::LightningTensor::TNCuda::Util {

enum class MPSStatus : uint32_t {
    BEGIN = 0,
    MPSInitNotSet = 0,
    MPSInitSet,
    MPSFinalizedNotSet,
    MPSFinalizedSet,
    END
};

/**
 * Utility function object to tell std::shared_ptr how to
 * release/destroy cutensornet objects.
 */
struct TNCudaHandleDeleter {
    void operator()(cutensornetHandle_t handle) const {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroy(handle));
    }
};

using SharedTNCudaHandle =
    std::shared_ptr<std::remove_pointer<cutensornetHandle_t>::type>;

/**
 * @brief Creates a SharedTNCudaHandle (a shared pointer to a cutensornetHandle)
 */
inline SharedTNCudaHandle make_shared_tncuda_handle() {
    cutensornetHandle_t h;
    PL_CUTENSORNET_IS_SUCCESS(cutensornetCreate(&h));
    return {h, TNCudaHandleDeleter()};
}

/**
 * @brief Returns the workspace size.
 *
 * @param tncuda_handle cutensornetHandle_t
 * @param workDesc cutensornetWorkspaceDescriptor_t
 *
 * @return std::size_t
 */
inline std::size_t
getWorkSpaceMemorySize(const cutensornetHandle_t &tncuda_handle,
                       cutensornetWorkspaceDescriptor_t &workDesc) {
    int64_t worksize{0};

    PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
        /* const cutensornetHandle_t */ tncuda_handle,
        /* cutensornetWorkspaceDescriptor_t */ workDesc,
        /* cutensornetWorksizePref_t */
        CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
        /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
        /*  int64_t * */ &worksize));

    // Ensure data is aligned by 256 bytes
    worksize += int64_t{256} - worksize % int64_t{256};

    return static_cast<std::size_t>(worksize);
}

/**
 * @brief Set memory for a workspace.
 *
 * @param tncuda_handle cutensornetHandle_t
 * @param workDesc cutensornet work space descriptor
 * @param scratchPtr Pointer to scratch memory
 * @param worksize Memory size of a work space
 */
inline void setWorkSpaceMemory(const cutensornetHandle_t &tncuda_handle,
                               cutensornetWorkspaceDescriptor_t &workDesc,
                               void *scratchPtr, std::size_t &worksize) {
    PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
        /* const cutensornetHandle_t */ tncuda_handle,
        /* cutensornetWorkspaceDescriptor_t */ workDesc,
        /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
        /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
        /* void *const */ scratchPtr,
        /* int64_t */ static_cast<int64_t>(worksize)));
}

/**
 * @brief Check if the wires are local.
 *
 * @param wires The wires to check.
 */
inline bool is_wires_local(const std::vector<std::size_t> &wires) {
    const std::size_t num_wires = wires.size();
    for (std::size_t i = 0; i < num_wires - 1; ++i) {
        if (wires[i + 1] - wires[i] != 1) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Create a queue of swap operations to be performed on the MPS.
 *
 * @param wires The target wires.
 *
 * @return A tuple containing the local target wires and the swap wire queue.
 */
inline auto create_swap_wire_pair_queue(const std::vector<std::size_t> &wires)
    -> std::tuple<std::vector<std::size_t>,
                  std::vector<std::vector<std::vector<std::size_t>>>> {
    PL_ABORT_IF_NOT(std::is_sorted(wires.begin(), wires.end()),
                    "The wires should be in descending order.");

    std::vector<std::vector<std::vector<std::size_t>>> swap_wires_queue;
    std::vector<std::size_t> local_wires;

    if (is_wires_local(wires)) {
        local_wires = wires;
    } else {
        const std::size_t num_wires = wires.size();

        const std::size_t fix_wire_pos = num_wires / std::size_t{2U};
        const std::size_t fixed_gate_wire_idx = wires[fix_wire_pos];

        local_wires.push_back(fixed_gate_wire_idx);

        int32_t left_wire_pos = fix_wire_pos - 1;
        int32_t right_wire_pos = fix_wire_pos + 1;

        while (left_wire_pos >= 0 ||
               right_wire_pos < static_cast<int32_t>(num_wires)) {
            std::vector<std::vector<std::size_t>> local_swap_wires_queue;
            if (left_wire_pos >= 0) {
                const std::size_t begin = wires[left_wire_pos];
                const std::size_t end =
                    wires[fix_wire_pos] - (fix_wire_pos - left_wire_pos);

                if (begin < end) {
                    for (std::size_t i = begin; i < end; i++) {
                        local_swap_wires_queue.emplace_back(
                            std::vector<std::size_t>{i, i + 1});
                    }
                    swap_wires_queue.emplace_back(local_swap_wires_queue);
                }

                std::size_t left_most_wire = local_wires[0] - 1;

                local_wires.insert(local_wires.begin(), left_most_wire);

                left_wire_pos--;
            }

            if (right_wire_pos < static_cast<int32_t>(num_wires)) {
                std::vector<std::vector<std::size_t>> local_swap_wires_queue;
                const std::size_t begin = wires[right_wire_pos];
                const std::size_t end =
                    wires[fix_wire_pos] + (right_wire_pos - fix_wire_pos);
                if (begin > end) {
                    for (std::size_t i = begin; i > end; i--) {
                        local_swap_wires_queue.emplace_back(
                            std::vector<std::size_t>{i, i - 1});
                    }
                    swap_wires_queue.emplace_back(local_swap_wires_queue);
                }

                std::size_t right_most_wire = local_wires.back() + 1;
                local_wires.push_back(right_most_wire);

                right_wire_pos++;
            }
        }
    }
    return {local_wires, swap_wires_queue};
}

} // namespace Pennylane::LightningTensor::TNCuda::Util
