// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "GateFunctors.hpp"
#include "Gates.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "TestHelpersWires.hpp"

/**
 * @file
 *  Tests for parametric gates functionality defined in the class
 * StateVectorKokkos.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Gates;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Functors;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Util;
using Pennylane::LightningKokkos::Util::getRealOfComplexInnerProduct;
} // namespace
/// @endcond
