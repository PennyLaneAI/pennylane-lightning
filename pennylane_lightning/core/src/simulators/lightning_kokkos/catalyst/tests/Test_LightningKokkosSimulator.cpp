// Copyright 2018-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "LightningKokkosSimulator.hpp"
#include "QuantumDevice.hpp"
#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Catalyst::Runtime::Simulator;
using namespace Pennylane::Util;
using LKSimulator = LightningKokkosSimulator;
using QDevice = Catalyst::Runtime::QuantumDevice;

GENERATE_DEVICE_FACTORY(LightningKokkosSimulator,
                        Catalyst::Runtime::Simulator::LightningKokkosSimulator);
} // namespace
/// @endcond
