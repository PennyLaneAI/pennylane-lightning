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
 * @file tncudaError.hpp
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
} // namespace
/// @endcond

#ifndef CUDA_UNSAFE

/**
 * @brief Macro that throws Exception from cuQuantum cutensornet failure error
 * codes.
 *
 * @param err cuQuantum cutensornet function error-code.
 */
#define PL_CUTENSORNET_IS_SUCCESS(err)                                         \
    PL_ABORT_IF_NOT(err == CUTENSORNET_STATUS_SUCCESS,                         \
                    Pennylane::LightningTensor::TNCuda::Util::                 \
                        GetCuTensorNetworkErrorString(err)                     \
                            .c_str())
#else
#define PL_CUTENSORNET_IS_SUCCESS(err)                                         \
    { static_cast<void>(err); }
#endif

namespace Pennylane::LightningTensor::TNCuda::Util {
static const std::string
GetCuTensorNetworkErrorString(const cutensornetStatus_t &err) {
    return cutensornetGetErrorString(err);
}
} // namespace Pennylane::LightningTensor::TNCuda::Util
  // LCOV_EXCL_STOP
