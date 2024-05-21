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
 * @file
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Tensor MPS class.
 */

#pragma once

#include <complex>
#include <cuda.h>
#include <cutensornet.h>
#include <type_traits>
#include <vector>

#include "MPSTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "ObservablesTNCudaOperator.hpp"
#include "TNCudaGateCache.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class couples with a state tensor to performs measurements.
 * Observables are defined by its operator(matrix), the observable class,
 * or through a string-based function dispatch.
 *
 * @tparam StateTensorT type of the state tensor to be measured.
 */
template <class StateTensorT> class Measurements {
  private:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using ComplexT = typename StateTensorT::ComplexT;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));

    StateTensorT &state_tensor_;

  public:
    explicit Measurements(StateTensorT &state_tensor)
        : state_tensor_(state_tensor){};

    /**
     * @brief Calculate expectation value for a general Observable represented
     * by an ObservableTNCudaOperator object.
     *
     * @param ob Observable operator.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(Pennylane::LightningTensor::TNCuda::Observables::Observable<
                StateTensorT> &ob) -> PrecisionT {
        auto tnoperator =
            ObservableTNCudaOperator<StateTensorT>(state_tensor_, ob);
        return state_tensor_.expval(tnoperator.getTNOperator()).real();
    }
};
} // namespace Pennylane::LightningTensor::TNCuda::Measures
