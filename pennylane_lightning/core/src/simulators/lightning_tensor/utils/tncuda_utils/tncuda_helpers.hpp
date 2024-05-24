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

} // namespace Pennylane::LightningTensor::TNCuda::Util
