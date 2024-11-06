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
 * @file MPSTNCuda.hpp
 * MPS class with cuTensorNet backend. Note that current implementation only
 * support the open boundary condition.
 */

#pragma once

#include <vector>

#include "DevTag.hpp"
#include "TNCudaBase.hpp"
#include "TensorCuda.hpp"
#include "TensornetBase.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda {

/**
 * @brief Managed memory Exact Tensor Network class using cutensornet high-level
 * APIs backed.
 *
 * @tparam Precision Floating-point precision type.
 */

template <class Precision>
class ExaTNCuda final : public TNCudaBase<Precision, ExaTNCuda<Precision>> {
  private:
    using BaseType = TNCudaBase<Precision, ExaTNCuda>;

  public:
    constexpr static auto method = "exatn";

    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  public:
    ExaTNCuda() = delete;

    explicit ExaTNCuda(const std::size_t numQubits) : BaseType(numQubits) {}

    explicit ExaTNCuda(const std::size_t numQubits, DevTag<int> dev_tag)
        : BaseType(numQubits, dev_tag) {}

    ~ExaTNCuda() = default;
};
} // namespace Pennylane::LightningTensor::TNCuda
