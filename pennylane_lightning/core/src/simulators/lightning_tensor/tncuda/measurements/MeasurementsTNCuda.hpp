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

#include <algorithm>
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

        DataBuffer<PrecisionT, int> d_output_probs(
            length, tensor_network_.getDevTag(), true);

        d_output_tensor.zeroInit();
        d_output_probs.zeroInit();

        auto stateModes = cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
            wires, tensor_network_.getNumQubits());

        std::vector<int32_t> projected_modes{};

        for (int32_t idx = 0;
             idx < static_cast<int32_t>(tensor_network_.getNumQubits());
             idx++) {
            auto it = std::find(stateModes.begin(), stateModes.end(), idx);
            if (it == stateModes.end()) {
                projected_modes.emplace_back(idx);
            }
        }

        std::vector<int64_t> projectedModeValues(projected_modes.size(), 0);

        if (projected_modes.size() == 0) {
            tensor_network_.get_state_tensor(d_output_tensor.getData(),
                                             d_output_tensor.getLength(), {},
                                             {}, numHyperSamples);
            getProbs_CUDA(d_output_tensor.getData(), d_output_probs.getData(),
                          length, static_cast<int>(thread_per_block),
                          tensor_network_.getDevTag().getStreamID());

        } else {
            PL_ABORT_IF(projected_modes.size() > 64,
                        "Number of projected modes is greater than 64 and the "
                        "value of projected_modes_size will exceed "
                        "std::numeric_limits<size_t>::max()");
            const std::size_t projected_modes_size = std::size_t(1U)
                                                     << projected_modes.size();

            DataBuffer<PrecisionT, int> tmp_probs(
                length, tensor_network_.getDevTag(), true);

            for (std::size_t idx = 0; idx < projected_modes_size; idx++) {
                for (std::size_t j = 0; j < projected_modes.size(); j++) {
                    projectedModeValues[j] = (idx >> j) & 1U;
                }

                tensor_network_.get_state_tensor(
                    d_output_tensor.getData(), length, projected_modes,
                    projectedModeValues, numHyperSamples);

                getProbs_CUDA(d_output_tensor.getData(), tmp_probs.getData(),
                              length, static_cast<int>(thread_per_block),
                              tensor_network_.getDevTag().getStreamID());

                // Copy the data to the output tensor
                scaleAndAdd_CUDA(PrecisionT{1.0}, tmp_probs.getData(),
                                 d_output_probs.getData(),
                                 tmp_probs.getLength(),
                                 tensor_network_.getDevTag().getDeviceID(),
                                 tensor_network_.getDevTag().getStreamID(),
                                 tensor_network_.getCublasCaller());
            }
        }

        // `10` here means `1024` elements to be calculated
        // LCOV_EXCL_START
        if (wires.size() > 10) {
            PrecisionT sum;

            asum_CUDA_device<PrecisionT>(
                d_output_probs.getData(), length,
                tensor_network_.getDevTag().getDeviceID(),
                tensor_network_.getDevTag().getStreamID(),
                tensor_network_.getCublasCaller(), &sum);

            PL_ABORT_IF(sum == 0.0, "Sum of probabilities is zero.");

            normalizeProbs_CUDA(d_output_probs.getData(), length, sum,
                                static_cast<int>(thread_per_block),
                                tensor_network_.getDevTag().getStreamID());

            d_output_probs.CopyGpuDataToHost(h_res.data(), h_res.size());
        } else {
            // LCOV_EXCL_STOP
            // This branch dispatches the calculation to the CPU for a small
            // number of wires. The CPU calculation is faster than the GPU
            // calculation for a small number of wires due to the overhead of
            // the GPU kernel launch.
            d_output_probs.CopyGpuDataToHost(h_res.data(), h_res.size());

            // TODO: OMP support
            PrecisionT sum =
                std::accumulate(h_res.begin(), h_res.end(), PrecisionT{0.0});

            PL_ABORT_IF(sum == 0.0, "Sum of probabilities is zero.");
            // TODO: OMP support
            for (std::size_t i = 0; i < length; i++) {
                h_res[i] /= sum;
            }
        }

        return h_res;
    }

    /**
     * @brief Utility method for samples.
     *
     * @param wires Wires can be a subset or the full system.
     * @param num_samples Number of samples
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is default as 1.
     *
     * @return std::vector<std::size_t> A 1-d array storing the samples.
     * Each sample has a length equal to the number of wires. Each sample can
     * be accessed using the stride `sample_id * num_wires`, where `sample_id`
     * is a number between `0` and `num_samples - 1`.
     */
    auto generate_samples(const std::vector<std::size_t> &wires,
                          const std::size_t num_samples,
                          const int32_t numHyperSamples = 1)
        -> std::vector<std::size_t> {
        std::vector<int64_t> samples(num_samples * wires.size());

        const std::vector<int32_t> modesToSample =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                wires, tensor_network_.getNumQubits());

        cutensornetStateSampler_t sampler;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateSampler(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetState_t */ tensor_network_.getQuantumState(),
            /* int32_t numModesToSample */ modesToSample.size(),
            /* const int32_t *modesToSample */ modesToSample.data(),
            /* cutensornetStateSampler_t * */ &sampler));

        // Configure the quantum circuit sampler
        const cutensornetSamplerAttributes_t samplerAttributes =
            CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetSamplerConfigure(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateSampler_t */ sampler,
            /* cutensornetSamplerAttributes_t */ samplerAttributes,
            /* const void *attributeValue */ &numHyperSamples,
            /* size_t attributeSize */ sizeof(numHyperSamples)));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetWorkspaceDescriptor_t * */ &workDesc));

        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        // Prepare the quantum circuit sampler for sampling
        PL_CUTENSORNET_IS_SUCCESS(cutensornetSamplerPrepare(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateSampler_t */ sampler,
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t unused as of v24.08 */ 0x0));

        std::size_t worksize =
            getWorkSpaceMemorySize(tensor_network_.getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device.\n");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<std::size_t> d_scratch(d_scratch_length,
                                          tensor_network_.getDevTag(), true);

        setWorkSpaceMemory(tensor_network_.getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetSamplerSample(
            /* const cutensornetHandle_t */ tensor_network_.getTNCudaHandle(),
            /* cutensornetStateSampler_t */ sampler,
            /* int64_t numShots */ num_samples,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ samples.data(),
            /* cudaStream_t unused as of v24.08  */ 0x0));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroySampler(sampler));

        std::vector<std::size_t> samples_size_t(samples.size());

        std::transform(samples.begin(), samples.end(), samples_size_t.begin(),
                       [](int64_t x) { return static_cast<std::size_t>(x); });
        return samples_size_t;
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
            /* std::size_t */ sizeof(numHyperSamples)));

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
            /* std::size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t [unused] */ 0x0));

        std::size_t worksize =
            getWorkSpaceMemorySize(tensor_network_.getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device.\n");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t) + 1;
        DataBuffer<std::size_t, int> d_scratch(
            d_scratch_length, tensor_network_.getDevTag(), true);

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
