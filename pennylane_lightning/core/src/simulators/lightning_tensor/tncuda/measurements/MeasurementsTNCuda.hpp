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
 * @file MeasurementsTNCuda.hpp
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Tensor class.
 */

#pragma once

#include <complex>
#include <cuComplex.h>
#include <cutensornet.h>
#include <vector>

#include "LinearAlg.hpp"
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
extern void getProbs_CUDA(cuComplex *state, float *probs, const int data_size,
                          const std::size_t thread_per_block,
                          cudaStream_t stream_id);
extern void getProbs_CUDA(cuDoubleComplex *state, double *probs,
                          const int data_size,
                          const std::size_t thread_per_block,
                          cudaStream_t stream_id);
extern void normalizeProbs_CUDA(float *probs, const int data_size,
                                const float sum,
                                const std::size_t thread_per_block,
                                cudaStream_t stream_id);
extern void normalizeProbs_CUDA(double *probs, const int data_size,
                                const double sum,
                                const std::size_t thread_per_block,
                                cudaStream_t stream_id);
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
    using CFP_t = typename TensorNetT::CFP_t;

    const TensorNetT &tensor_network_;

  public:
    explicit MeasurementsTNCuda(const TensorNetT &tensor_network)
        : tensor_network_(tensor_network){};

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @tparam thread_per_block Number of threads per block in the CUDA kernel
     * and is default as `256`. `256` is chosen as a default value because it is
     * a balance of warp size and occupancy. Note that this number is not
     * optimal for all cases and may need to be adjusted based on the specific
     * use case, especially the number of elements in the subset is small.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @param numHyperSamples Number of hyper samples to be used in the
     * calculation and is default as 1.
     *
     * @return Floating point std::vector with probabilities.
     */
    template <std::size_t thread_per_block = 256>
    auto probs(const std::vector<std::size_t> &wires,
               const int32_t numHyperSamples = 1) -> std::vector<PrecisionT> {
        PL_ABORT_IF_NOT(std::is_sorted(wires.begin(), wires.end()),
                        "Invalid wire indices order. Please ensure that the "
                        "wire indices are in the ascending order.");

        const std::size_t length = std::size_t{1} << wires.size();

        std::vector<PrecisionT> h_res(length, 0.0);

        DataBuffer<CFP_t, int> d_output_tensor(
            length, tensor_network_.getDevTag(), true);

        d_output_tensor.zeroInit();

        DataBuffer<PrecisionT, int> d_output_probs(
            length, tensor_network_.getDevTag(), true);

        tensor_network_.get_state_tensor(d_output_tensor.getData(),
                                         d_output_tensor.getLength(), wires,
                                         numHyperSamples);

        getProbs_CUDA(d_output_tensor.getData(), d_output_probs.getData(),
                      length, static_cast<int>(thread_per_block),
                      tensor_network_.getDevTag().getStreamID());

        PrecisionT sum;

        asum_CUDA_device<PrecisionT>(d_output_probs.getData(), length,
                                     tensor_network_.getDevTag().getDeviceID(),
                                     tensor_network_.getDevTag().getStreamID(),
                                     tensor_network_.getCublasCaller(), &sum);

        PL_ABORT_IF(sum == 0.0, "Sum of probabilities is zero.");

        normalizeProbs_CUDA(d_output_probs.getData(), length, sum,
                            static_cast<int>(thread_per_block),
                            tensor_network_.getDevTag().getStreamID());

        d_output_probs.CopyGpuDataToHost(h_res.data(), h_res.size());

        return h_res;
    }

    /**
     * @brief Calculate var value for a general ObservableTNCuda Observable.
     *
     * Current implementation ensure that only one cutensornetNetworkOperator_t
     * object is attached to the circuit.
     *
     * @param obs An Observable object.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is default as 1.
     */
    auto var(ObservableTNCuda<TensorNetT> &obs,
             const int32_t numHyperSamples = 1) -> PrecisionT {
        auto expectation_val =
            expval(obs, numHyperSamples); // The cutensornetNetworkOperator_t
                                          // object created in expval() will be
                                          // released after the function call.

        const bool val_cal = true;
        auto tnObs2Operator =
            ObservableTNCudaOperator<TensorNetT>(tensor_network_, obs, val_cal);
        auto expectation_squared_obs =
            expval_(tnObs2Operator.getTNOperator(), numHyperSamples);

        return expectation_squared_obs - expectation_val * expectation_val;
    }

    /**
     * @brief Calculate expectation value for a general ObservableTNCuda
     * Observable.
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

        return expval_(tnoperator.getTNOperator(), numHyperSamples);
    }

  private:
    /**
     * @brief Calculate expectation value for a cutensornetNetworkOperator_t
     * object.
     *
     * @param tnoperator A cutensornetNetworkOperator_t object.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is default as 1.
     *
     * @return Expectation value with respect to the given
     * cutensornetNetworkOperator_t object.
     */

    auto expval_(cutensornetNetworkOperator_t tnoperator,
                 const int32_t numHyperSamples) -> PrecisionT {
        ComplexT expectation_val{0.0, 0.0};
        ComplexT state_norm2{0.0, 0.0};

        cutensornetStateExpectation_t expectation;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateExpectation(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetState_t */ tensor_network_.getQuantumState(),
            /* cutensornetNetworkOperator_t */ tnoperator,
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
