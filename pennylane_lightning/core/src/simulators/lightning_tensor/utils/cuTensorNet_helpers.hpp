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

/**
 * @file cudaTensorNet_helpers.hpp
 */

#pragma once
#include <cutensornet.h>
#include <memory>
#include <utility>

#include "cuTensorNetError.hpp"

namespace Pennylane::LightningTensor::Util {

/**
 * Utility function object to tell std::shared_ptr how to
 * release/destroy cuTensorNet objects.
 */
struct handleDeleter {
    void operator()(cutensornetHandle_t handle) const {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroy(handle));
    }
};

using SharedCutnHandle =
    std::shared_ptr<std::remove_pointer<cutensornetHandle_t>::type>;

/**
 * @brief Creates a SharedCusvHandle (a shared pointer to a custatevecHandle)
 */
inline SharedCutnHandle make_shared_cutn_handle() {
    cutensornetHandle_t h;
    PL_CUTENSORNET_IS_SUCCESS(cutensornetCreate(&h));
    return {h, handleDeleter()};
}
} // namespace Pennylane::LightningTensor::Util