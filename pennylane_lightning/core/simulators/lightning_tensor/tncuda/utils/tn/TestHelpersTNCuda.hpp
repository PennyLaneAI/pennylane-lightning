// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
/**
 * @file
 * This file defines the necessary functionality to test over LTensor.
 */
#include "DevTag.hpp"
#include "ExactTNCuda.hpp"
#include "MPSTNCuda.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace

namespace Pennylane::LightningTensor::TNCuda::Util {
template <class MPS> struct MPSToName;

template <> struct MPSToName<MPSTNCuda<float>> {
    constexpr static auto name = "MPSTNCuda<float>";
};
template <> struct MPSToName<MPSTNCuda<double>> {
    constexpr static auto name = "MPSTNCuda<double>";
};

template <typename TNDevice_T>
std::unique_ptr<TNDevice_T> createTNState(std::size_t num_qubits,
                                          std::size_t maxExtent) {
    DevTag<int> dev_tag{0, 0};
    if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                  std::is_same_v<TNDevice_T, MPSTNCuda<float>>) {
        // Create the object for MPSTNCuda
        return std::make_unique<TNDevice_T>(num_qubits, maxExtent, dev_tag);
    } else {
        // Create the object for ExactTNCuda
        return std::make_unique<TNDevice_T>(num_qubits, dev_tag);
    }
}

template <typename TNDevice_T>
inline void
tn_state_append_mps_final_state(std::unique_ptr<TNDevice_T> const &tn_state) {
    if constexpr (std::is_same_v<TNDevice_T, MPSTNCuda<double>> ||
                  std::is_same_v<TNDevice_T, MPSTNCuda<float>>) {
        PL_ABORT_IF(
            tn_state->getMethod() != "mps",
            "The method append_mps_final_state is exclusive of MPS TensorNet");
        tn_state->append_mps_final_state();
    }
}

using TestTNBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>,
                              ExactTNCuda<float>, ExactTNCuda<double>>;
using TestMPSBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>>;

} // namespace Pennylane::LightningTensor::TNCuda::Util
/// @endcond
