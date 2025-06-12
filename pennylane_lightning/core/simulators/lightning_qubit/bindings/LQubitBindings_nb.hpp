// Copyright 2025 Xanadu Quantum Technologies Inc.

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
 * @file LQubitBindings_nb.hpp
 * Defines lightning.qubit specific operations to export to Python using
 * Nanobind.
 */

#pragma once
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "StateVectorLQubitManaged.hpp"
#include "TypeList.hpp"

namespace nb = nanobind;

namespace Pennylane::LightningQubit {

/**
 * @brief Define StateVector backends for lightning.qubit
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorLQubitManaged<float>,
                              StateVectorLQubitManaged<double>, void>;

} // namespace Pennylane::LightningQubit
