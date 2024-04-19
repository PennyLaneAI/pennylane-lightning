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

// Adapted from JET: https://github.com/XanaduAI/jet.git
// and from Lightning: https://github.com/PennylaneAI/pennylane-lightning.git

/**
 * @file cuStateVecError.hpp
 * Defines macros that throws Exception from cuStateVec failure error codes.
 */

#pragma once
#include <string>
#include <string_view>

#include <custatevec.h>

#include "Error.hpp"
#include "Util.hpp"

// LCOV_EXCL_START
/// @cond DEV
namespace {
using namespace Pennylane::Util;
}
/// @endcond

#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from cuStateVec failure error codes.
 *
 * @param err cuStateVec function error-code.
 */
#define PL_CUSTATEVEC_IS_SUCCESS(err)                                          \
    PL_ABORT_IF_NOT(                                                           \
        err == CUSTATEVEC_STATUS_SUCCESS,                                      \
        Pennylane::LightningGPU::Util::GetCuStateVecErrorString(err).c_str())

#else
#define PL_CUSTATEVEC_IS_SUCCESS(err)                                          \
    { static_cast<void>(err); }
#endif

namespace Pennylane::LightningGPU::Util {
static const std::string
GetCuStateVecErrorString(const custatevecStatus_t &err) {
    using namespace std::string_literals;

    switch (err) {
    case CUSTATEVEC_STATUS_SUCCESS:
        return "No errors"s;
    case CUSTATEVEC_STATUS_NOT_INITIALIZED:
        return "custatevec not initialized"s;
    case CUSTATEVEC_STATUS_ALLOC_FAILED:
        return "custatevec memory allocation failed"s;
    case CUSTATEVEC_STATUS_INVALID_VALUE:
        return "custatevec invalid value"s;
    case CUSTATEVEC_STATUS_ARCH_MISMATCH:
        return "custatevec CUDA device architecture mismatch"s;
    case CUSTATEVEC_STATUS_EXECUTION_FAILED:
        return "custatevec execution failed"s;
    case CUSTATEVEC_STATUS_INTERNAL_ERROR:
        return "custatevec internal error"s;
    case CUSTATEVEC_STATUS_NOT_SUPPORTED:
        return "custatevec unsupported operation/device"s;
    case CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE:
        return "custatevec insufficient memory for gate-application workspace"s;
    case CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED:
        return "custatevec sampler not preprocessed"s;
    case CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR:
        return "custatevec no device allocator"s;
    case CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR:
        return "custatevec device allocator error"s;
    case CUSTATEVEC_STATUS_COMMUNICATOR_ERROR:
        return "custatevec communicator failure"s;
    case CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED:
        return "custatevec dynamic library load failure"s;
    default:
        return "custatevec status not found. Error code="s +
               std::to_string(err);
    }
}
} // namespace Pennylane::LightningGPU::Util
// LCOV_EXCL_STOP
