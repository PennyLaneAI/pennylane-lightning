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
 * represented by a Lightning Tensor class.
 */

#pragma once

#include <complex>
#include <cutensornet.h>
#include <vector>

#include "MPSTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "ObservablesTNCudaOperator.hpp"

#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Measures {
/**
 * @brief ObservablesTNCuda's Measurement Class.
 *
 * This class couples with a tensor network to perform measurements.
 * Observables are defined in the observable class.
 *
 * @tparam TensorNetT type of the tensor network to be measured.
 */
template <class TensorNetT> class MeasurementsTNCuda {
  private:
    using PrecisionT = typename TensorNetT::PrecisionT;
    using ComplexT = typename TensorNetT::ComplexT;

    const TensorNetT &tensor_network_;

  public:
    explicit MeasurementsTNCuda(const TensorNetT &tensor_network)
        : tensor_network_(tensor_network){};

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param obs An Observable object.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is default as 1.
     *
     * @return Expectation value with respect to the given observable.
     */
    auto expval(ObservableTNCuda<TensorNetT> &obs,
                const int32_t numHyperSamples = 1) -> PrecisionT {
        auto tnoperator =
            ObservableTNCudaOperator<TensorNetT>(tensor_network_, obs);

        ComplexT expectation_val{0.0, 0.0};
        ComplexT state_norm2{0.0, 0.0};

        cutensornetStateExpectation_t expectation;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateExpectation(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetState_t */ tensor_network_.getQuantumState(),
            /* cutensornetNetworkOperator_t */ tnoperator.getTNOperator(),
            /* cutensornetStateExpectation_t * */ &expectation));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationConfigure(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetExpectationAttributes_t */
            CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES,
            /* const void * */ &numHyperSamples,
            /* size_t */ sizeof(numHyperSamples)));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetWorkspaceDescriptor_t * */ &workDesc));

        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        // Prepare the specified quantum circuit expectation value for
        // computation
        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationPrepare(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t [unused] */ 0x0));

        std::size_t worksize =
            getWorkSpaceMemorySize(tensor_network_.getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device.\n");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<size_t, int> d_scratch(d_scratch_length,
                                          tensor_network_.getDevTag(), true);

        setWorkSpaceMemory(tensor_network_.getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationCompute(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* void* */ static_cast<void *>(&expectation_val),
            /* void* */ static_cast<void *>(&state_norm2),
            /* cudaStream_t unused */ 0x0));

        expectation_val /= state_norm2;

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyExpectation(expectation));

        return expectation_val.real();
    }
};
} // namespace Pennylane::LightningTensor::TNCuda::Measures
