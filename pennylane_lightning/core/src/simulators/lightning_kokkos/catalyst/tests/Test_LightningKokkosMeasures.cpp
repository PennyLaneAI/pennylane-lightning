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

#include <random>

#include "CacheManager.hpp"
#include "LightningKokkosSimulator.hpp"
#include "QuantumDevice.hpp"
#include "Types.h"
#include "Utils.hpp"
#include "catch2/catch.hpp"
#include "cmath"

/// @cond DEV
namespace {
// MemRef type definition (Helper)
template <typename T, std::size_t R> struct MemRefT {
    T *data_allocated;
    T *data_aligned;
    std::size_t offset;
    std::size_t sizes[R];
    std::size_t strides[R];
};
using namespace Catalyst::Runtime::Simulator;
using LKSimulator = LightningKokkosSimulator;
} // namespace
/// @endcond
