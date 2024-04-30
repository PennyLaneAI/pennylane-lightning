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
 * @file cutnError.hpp
 * Defines macros that throws Exception from cutensornet failure error codes.
 */

#pragma once
#include "Error.hpp"
#include "Util.hpp"
#include <cutensornet.h>
#include <string>

// LCOV_EXCL_START
/// @cond DEV
namespace {
using namespace Pennylane::Util;
}
/// @endcond

#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from cuQuantum cutensornet failure error
 * codes.
 *
 * @param err cuQuantum cutensornet function error-code.
 */
#define PL_CUTENSORNET_IS_SUCCESS(err)                                         \
    PL_ABORT_IF_NOT(                                                           \
        err == CUTENSORNET_STATUS_SUCCESS,                                     \
        Pennylane::LightningTensor::Cutn::Util::GetCuTensorNetworkErrorString( \
            err)                                                               \
            .c_str())
#else
#define PL_CUTENSORNET_IS_SUCCESS                                              \
    { static_cast<void>(err); }
#endif

namespace Pennylane::LightningTensor::Cutn::Util {
static const std::string
GetCuTensorNetworkErrorString(const cutensornetStatus_t &err) {
    using namespace std::string_literals;
    switch (err) {
    case CUTENSORNET_STATUS_SUCCESS:
        return "No errors"s;
    case CUTENSORNET_STATUS_NOT_INITIALIZED:
        return "cutensornet not initialized"s;
    case CUTENSORNET_STATUS_ALLOC_FAILED:
        return "cutensornet memory allocation failed"s;
    case CUTENSORNET_STATUS_INVALID_VALUE:
        return "cutensornet invalid value"s;
    case CUTENSORNET_STATUS_ARCH_MISMATCH:
        return "cutensornet CUDA device architecture mismatch"s;
    case CUTENSORNET_STATUS_MAPPING_ERROR:
        return "cutensornet GPU memory space failed"s;
    case CUTENSORNET_STATUS_EXECUTION_FAILED:
        return "cutensornet execute error"s;
    case CUTENSORNET_STATUS_INTERNAL_ERROR:
        return "cutensornet internal error"s;
    case CUTENSORNET_STATUS_NOT_SUPPORTED:
        return "cutensornet unsupported operation/device"s;
    case CUTENSORNET_STATUS_LICENSE_ERROR:
        return "cutensornet license error"s;
    case CUTENSORNET_STATUS_CUBLAS_ERROR:
        return "cutensornet call to cublas failed"s;
    case CUTENSORNET_STATUS_CUDA_ERROR:
        return "cutensornet unknown CUDA error"s;
    case CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE:
        return "cutensornet provided workspace was insufficient"s;
    case CUTENSORNET_STATUS_INSUFFICIENT_DRIVER:
        return "cutensornet driver version is insufficient"s;
    case CUTENSORNET_STATUS_IO_ERROR:
        return "cutensornet IO error"s;
    case CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH:
        return "cutensornet incompatible cuTensor library"s;
    case CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR:
        return "cutensornet mempool is not set"s;
    case CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED:
        return "cutensornet all hyper samples failed for one or more errors "
               "please enable LOGs via export CUTENSORNET_LOG_LEVEL= > 1 for "
               "details"s;
    case CUTENSORNET_STATUS_CUSOLVER_ERROR:
        return "cutensornet cusolver failed"s;
    case CUTENSORNET_STATUS_DEVICE_ALLOCATOR_ERROR:
        return "cutensornet operation with the device memory pool failed"s;
    case CUTENSORNET_STATUS_DISTRIBUTED_FAILURE:
        return "cutensornet distributed communication service failure"s;
    case CUTENSORNET_STATUS_INTERRUPTED:
        return "cutensornet operation interruption"s;
    default:
        return "cutensornet status not found. Error code="s +
               std::to_string(err);
    }
}
} // namespace Pennylane::LightningTensor::Cutn::Util
  // LCOV_EXCL_STOP
  