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
 * This file defines the necessary functionality to test over LTensor MPS.
 */
#include "MPSCutn.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::Cutn;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::Cutn::Util {
template <class MPS> struct MPSToName;

template <> struct MPSToName<MPSCutn<float>> {
    constexpr static auto name = "MPSCutn<float>";
};
template <> struct MPSToName<MPSCutn<double>> {
    constexpr static auto name = "MPSCutn<double>";
};

using TestMPSBackends =
    Pennylane::Util::TypeList<MPSCutn<float>, MPSCutn<double>, void>;
} // namespace Pennylane::LightningTensor::Cutn::Util
