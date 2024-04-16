// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Adapted from JET: https://github.com/XanaduAI/jet.git
// and from Lightning: https://github.com/PennylaneAI/pennylane-lightning.git
/**
 * @file cuTensorNetError.hpp
 */

#pragma once
#include "Error.hpp"
#include "Util.hpp"
#include <cutensornet.h>
#include <string>
// LCOV_EXCL_START
namespace {
using namespace Pennylane::Util;
}

#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from cuQuantum cuTensorNet failure error
 * codes.
 *
 * @param err cuQuantum cuTensorNet function error-code.
 */
#define PL_CUTENSORNET_IS_SUCCESS(err)                                         \
    PL_ABORT_IF_NOT(                                                           \
        err == CUTENSORNET_STATUS_SUCCESS,                                     \
        Pennylane::LightningTensor::Util::GetCuTensorNetworkErrorString(err)   \
            .c_str())
#else
#define PL_CUTENSORNET_IS_SUCCESS                                              \
    { static_cast<void>(err); }
#endif

namespace Pennylane::LightningTensor::Util {
static const std::string
GetCuTensorNetworkErrorString(const cutensornetStatus_t &err) {
    std::string result;
    switch (err) {
    case CUTENSORNET_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CUTENSORNET_STATUS_NOT_INITIALIZED:
        result = "cutensornet not initialized";
        break;
    case CUTENSORNET_STATUS_ALLOC_FAILED:
        result = "cutensornet memory allocation failed";
        break;
    case CUTENSORNET_STATUS_INVALID_VALUE:
        result = "cutensornet invalid value";
        break;
    case CUTENSORNET_STATUS_ARCH_MISMATCH:
        result = "cutensornet CUDA device architecture mismatch";
        break;
    case CUTENSORNET_STATUS_MAPPING_ERROR:
        result = "cutensornet GPU memory space failed";
        break;
    case CUTENSORNET_STATUS_EXECUTION_FAILED:
        result = "cutensornet execute error";
        break;
    case CUTENSORNET_STATUS_INTERNAL_ERROR:
        result = "cutensornet internal error";
        break;
    case CUTENSORNET_STATUS_NOT_SUPPORTED:
        result = "cutensornet unsupported operation/device";
        break;
    case CUTENSORNET_STATUS_LICENSE_ERROR:
        result = "cutensornet license error";
        break;
    case CUTENSORNET_STATUS_CUBLAS_ERROR:
        result = "cutensornet call to cublas failed";
        break;
    case CUTENSORNET_STATUS_CUDA_ERROR:
        result = "cutensornet unknown CUDA error";
        break;
    case CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE:
        result = "cutensornet provided workspace was insufficient";
        break;
    case CUTENSORNET_STATUS_INSUFFICIENT_DRIVER:
        result = "cutensornet driver version is insufficient";
        break;
    case CUTENSORNET_STATUS_IO_ERROR:
        result = "cutensornet IO error";
        break;
    case CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH:
        result = "cutensornet incompatible cuTensor library";
        break;
    case CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR:
        result = "cutensornet mempool is not set";
        break;
    case CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED:
        result = "cutensornet all hyper samples failed for one or more errors "
                 "please enable LOGs via export CUTENSORNET_LOG_LEVEL= > 1 for "
                 "details";
        break;
    case CUTENSORNET_STATUS_CUSOLVER_ERROR:
        result = "cutensornet cusolver failed";
        break;
    case CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR:
        result = "cutensornet operation with the device memory pool failed";
        break;
    case CUTENSORNET_STATUS_DISTRIBUTED_FAILURE:
        result = "cutensornet distributed communication service failure";
        break;
    case CUTENSORNET_STATUS_INTERRUPTED:
        result = "cutensornet operation interruption";
        break;
    default:
        result =
            "cutensornet status not found. Error code=" + std::to_string(err);
    }
    return result;
}
} // namespace Pennylane::LightningTensor::Util
  // LCOV_EXCL_STOP