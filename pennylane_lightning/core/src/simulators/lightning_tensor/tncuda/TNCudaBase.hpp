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
 * @file TNCudaBase.hpp
 * Base class for cuTensorNet-backed tensor networks.
 */

#pragma once

#include <complex>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "LinearAlg.hpp"
#include "TNCudaGateCache.hpp"
#include "TensorBase.hpp"
#include "TensorCuda.hpp"
#include "TensornetBase.hpp"
#include "cuda_helpers.hpp"
#include "tncudaError.hpp"
#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::TNCuda {
/**
 * @brief CRTP-enabled base class for cuTensorNet backends.
 *
 * @tparam PrecisionT Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */
template <class PrecisionT, class Derived>
class TNCudaBase : public TensornetBase<PrecisionT, Derived> {
  private:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using ComplexT = std::complex<PrecisionT>;
    using BaseType = TensornetBase<PrecisionT, Derived>;
    SharedTNCudaHandle handle_;
    SharedCublasCaller cublascaller_;
    cudaDataType_t typeData_;
    DevTag<int> dev_tag_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

    std::shared_ptr<TNCudaGateCache<PrecisionT>> gate_cache_;
    std::set<int64_t> gate_ids_;

    std::vector<std::size_t> identiy_gate_ids_;

  public:
    TNCudaBase() = delete;

    // TODO: Add method to the constructor to all user to select methods at
    // runtime in the C++ layer
    explicit TNCudaBase(const std::size_t numQubits, int device_id = 0,
                        cudaStream_t stream_id = 0)
        : BaseType(numQubits), handle_(make_shared_tncuda_handle()),
          cublascaller_(make_shared_cublas_caller()),
          dev_tag_({device_id, stream_id}),
          gate_cache_(std::make_shared<TNCudaGateCache<PrecisionT>>(dev_tag_)) {
        // TODO this code block could be moved to base class and need to revisit
        // when working on copy ctor
        PL_ABORT_IF(numQubits < 2,
                    "The number of qubits should be greater than 1.");

        if constexpr (std::is_same_v<PrecisionT, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */
            static_cast<int32_t>(BaseType::getNumQubits()),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(BaseType::getQubitDims().data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));
    }

    // TODO: Add method to the constructor to all user to select methods at
    // runtime in the C++ layer
    explicit TNCudaBase(const std::size_t numQubits, DevTag<int> dev_tag)
        : BaseType(numQubits), handle_(make_shared_tncuda_handle()),
          cublascaller_(make_shared_cublas_caller()), dev_tag_(dev_tag),
          gate_cache_(std::make_shared<TNCudaGateCache<PrecisionT>>(dev_tag_)) {
        // TODO this code block could be moved to base class and need to revisit
        // when working on copy ctor
        PL_ABORT_IF(numQubits < 2,
                    "The number of qubits should be greater than 1.");
        if constexpr (std::is_same_v<PrecisionT, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */
            static_cast<int32_t>(BaseType::getNumQubits()),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(BaseType::getQubitDims().data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));
    }

    ~TNCudaBase() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    /**
     * @brief Get the CUDA data type.
     *
     * @return cudaDataType_t
     */
    [[nodiscard]] auto getCudaDataType() const -> cudaDataType_t {
        return typeData_;
    }

    /**
     * @brief Get the cutensornet handle that the object is using.
     *
     * @return cutensornetHandle_t
     */
    [[nodiscard]] auto getTNCudaHandle() const -> cutensornetHandle_t {
        return handle_.get();
    }

    /**
     * @brief Access the CublasCaller the object is using.
     *
     * @return a reference to the object's CublasCaller object.
     */
    auto getCublasCaller() const -> const CublasCaller & {
        return *cublascaller_;
    }

    /**
     * @brief Get the quantum state pointer.
     *
     * @return cutensornetState_t
     */
    [[nodiscard]] auto getQuantumState() const -> cutensornetState_t {
        return quantumState_;
    };

    /**
     * @brief Get device and Cuda stream information (device ID and the
     * associated Cuda stream ID).
     *
     * @return DevTag
     */
    [[nodiscard]] auto getDevTag() const -> const DevTag<int> & {
        return dev_tag_;
    }

    /**
     * @brief Append multiple gates to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     * @param ops_params Vector of gate parameters.
     */
    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<std::size_t>> &ops_wires,
                    const std::vector<bool> &ops_adjoint,
                    const std::vector<std::vector<PrecisionT>> &ops_params) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_adjoint[i], ops_params[i]);
        }
    }

    /**
     * @brief Append multiple gate tensors to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<std::size_t>> &ops_wires,
                         const std::vector<bool> &ops_adjoint) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_adjoint[i], {});
        }
    }

    /**
     * @brief Append a single controlled gate tensor to the compute graph.
     *
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph. The controlled gate should be
     * immutable as v24.08.
     *
     * @param baseOpName Base gate's name.
     * @param controlled_wires Controlled wires for the gate.
     * @param controlled_values Controlled values for the gate.
     * @param targetWires Target wires for the gate.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix for custom gates.
     */
    void
    applyControlledOperation(const std::string &baseOpName,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &targetWires,
                             bool adjoint = false,
                             const std::vector<PrecisionT> &params = {0.0},
                             const std::vector<ComplexT> &gate_matrix = {}) {
        // TODO: Need to revisit this line of code once `cutensornet` supports
        // multi-target wire controlled gates
        PL_ABORT_IF_NOT(targetWires.size() == 1,
                        "Unsupported controlled gate: cutensornet only "
                        "supports 1-wire target controlled gates");

        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        int64_t dummy_id = gate_ids_.empty() ? 1 : *gate_ids_.rbegin() + 1;

        if (gate_matrix.empty()) {
            gate_cache_->add_gate(dummy_id, baseOpName, par, adjoint);
        } else {
            auto gate_key = std::make_pair(baseOpName, par);
            std::vector<CFP_t> matrix_cu =
                cuUtil::complexToCu<ComplexT>(gate_matrix);
            gate_cache_->add_gate(dummy_id, gate_key, matrix_cu, adjoint);
        }

        int64_t id;

        std::vector<int32_t> controlledModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                controlled_wires, BaseType::getNumQubits());

        std::vector<int64_t> controlled_values_int64(controlled_values.size());
        std::transform(controlled_values.begin(), controlled_values.end(),
                       controlled_values_int64.begin(),
                       [](bool val) { return static_cast<int64_t>(val); });

        std::vector<int32_t> targetModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                targetWires, BaseType::getNumQubits());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyControlledTensorOperator(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* int32_t numControlModes */ controlled_wires.size(),
            /* const int32_t * stateControlModes */ controlledModes.data(),
            /* const int64_t *stateControlValues*/
            controlled_values_int64.data(),
            /* int32_t numTargetModes */ targetWires.size(),
            /* const int32_t * stateTargetModes */ targetModes.data(),
            /* void * */
            static_cast<void *>(gate_cache_->get_gate_device_ptr(dummy_id)),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 1,
            /* const int32_t adjoint */ 0,
            /* const int32_t unitary */ 1,
            /* int64_t tensorId* */ &id));

        if (dummy_id != id) {
            gate_cache_->update_key(dummy_id, id);
        }

        gate_ids_.insert(id);
    }

    /**
     * @brief Append a single gate tensor to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param opName Gate's name.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix for custom gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool adjoint = false,
                        const std::vector<PrecisionT> &params = {0.0},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        // TODO: Need to revisit this line of code for the exact TN backend.
        //  We should be able to turn on/ skip this check based on the backend,
        //  if(getMethod() == "mps") { ... }
        PL_ABORT_IF(
            wires.size() > 2,
            "Unsupported gate: MPS method only supports 1, 2-wires gates");

        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        int64_t dummy_id = gate_ids_.empty() ? 1 : *gate_ids_.rbegin() + 1;

        if (gate_matrix.empty()) [[likely]] {
            gate_cache_->add_gate(dummy_id, opName, par, adjoint);
        } else [[unlikely]] {
            auto gate_key = std::make_pair(opName, par);
            std::vector<CFP_t> matrix_cu =
                cuUtil::complexToCu<ComplexT>(gate_matrix);
            gate_cache_->add_gate(dummy_id, gate_key, matrix_cu, adjoint);
        }

        int64_t id;

        std::vector<int32_t> stateModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                wires, BaseType::getNumQubits());

        //  Note `adjoint` in the cutensornet context indicates whether or not
        //  all tensor elements of the tensor operator will be complex
        //  conjugated. `adjoint` in the following API is not equivalent to
        //  `inverse` in the lightning context
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyTensorOperator(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* int32_t numStateModes */ stateModes.size(),
            /* const int32_t * stateModes */ stateModes.data(),
            /* void * */
            static_cast<void *>(gate_cache_->get_gate_device_ptr(dummy_id)),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 0,
            /* const int32_t adjoint */ 0,
            /* const int32_t unitary */ 1,
            /* int64_t * */ &id));

        if (dummy_id != id) {
            gate_cache_->update_key(dummy_id, id);
        }

        // one time initialization of the identity gate id
        if (identiy_gate_ids_.empty() && opName == "Identity") {
            identiy_gate_ids_.push_back(static_cast<std::size_t>(id));
        }

        gate_ids_.insert(id);
    }

    /**
     * @brief Get the state vector representation of a tensor network.
     *
     * @param host_data Pointer to the host memory for state tensor data.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_state_tensor(ComplexT *host_data,
                          const int32_t numHyperSamples = 1) {
        std::vector<int32_t> projected_modes{};
        std::vector<int64_t> projected_mode_values{};

        const std::size_t length = std::size_t{1} << BaseType::getNumQubits();

        DataBuffer<CFP_t, int> d_output_tensor(length, getDevTag(), true);

        get_accessor_(d_output_tensor.getData(), length, projected_modes,
                      projected_mode_values, numHyperSamples);

        d_output_tensor.CopyGpuDataToHost(host_data, length);
    }

    /**
     * @brief Get a slice of the full state tensor.
     *
     * @param tensor_data Pointer to the device memory for state tensor data.
     * @param tensor_data_size Size of the state tensor data.
     * @param projected_modes Projected modes to get the state tensor for.
     * @param projected_mode_values Values of the projected modes.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_state_tensor(CFP_t *tensor_data,
                          const std::size_t tensor_data_size,
                          const std::vector<int32_t> &projected_modes,
                          const std::vector<int64_t> &projected_mode_values,
                          const int32_t numHyperSamples = 1) const {
        get_accessor_(tensor_data, tensor_data_size, projected_modes,
                      projected_mode_values, numHyperSamples);
    }

  private:
    /**
     * @brief Get accessor of a state tensor
     *
     * @param tensor_data Pointer to the device memory for state tensor data.
     * @param tensor_data_size Size of the tensor data.
     * @param projected_modes Projected modes to get the state tensor for.
     * @param projected_mode_values Values of the projected modes.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_accessor_(CFP_t *tensor_data, const std::size_t tensor_data_size,
                       const std::vector<int32_t> &projected_modes,
                       const std::vector<int64_t> &projected_mode_values,
                       const int32_t numHyperSamples = 1) const {
        cutensornetStateAccessor_t accessor;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateAccessor(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* int32_t numProjectedModes */
            static_cast<int32_t>(projected_modes.size()),
            /* const int32_t *projectedModes */ projected_modes.data(),
            /* const int64_t *amplitudesTensorStrides */ nullptr,
            /* cutensornetStateAccessor_t *tensorNetworkAccessor*/ &accessor));

        // Configure the computation
        const cutensornetAccessorAttributes_t accessor_attribute =
            CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorConfigure(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetStateAccessor_t */ accessor,
            /* cutensornetAccessorAttributes_t */ accessor_attribute,
            /* const void * */ &numHyperSamples,
            /* std::size_t */ sizeof(numHyperSamples)));

        // prepare the computation
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(getTNCudaHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorPrepare(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetStateAccessor_t*/ accessor,
            /* std::size_t */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t unused as of v24.03 */ 0x0));

        // Allocate workspace buffer
        std::size_t worksize =
            getWorkSpaceMemorySize(getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t);
        DataBuffer<std::size_t, int> d_scratch(d_scratch_length, getDevTag(),
                                               true);

        setWorkSpaceMemory(getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        // compute the specified slice of the quantum circuit amplitudes tensor
        ComplexT stateNorm2{0.0, 0.0};
        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorCompute(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetStateAccessor_t */ accessor,
            /* const int64_t * projectedModeValues */
            projected_mode_values.data(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* void *amplitudesTensor*/
            static_cast<void *>(tensor_data),
            /* void *stateNorm */ static_cast<void *>(&stateNorm2),
            /* cudaStream_t cudaStream */ 0x0));

        PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(getDevTag().getStreamID()));

        const ComplexT scale_scalar = ComplexT{1.0, 0.0} / stateNorm2;

        CFP_t scale_scalar_cu{scale_scalar.real(), scale_scalar.imag()};

        scaleC_CUDA<CFP_t, CFP_t>(scale_scalar_cu, tensor_data,
                                  tensor_data_size, getDevTag().getDeviceID(),
                                  getDevTag().getStreamID(), getCublasCaller());

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyAccessor(accessor));
    }

  protected:
    /**
     * @brief Dummy tensor operator update to allow multiple calls of
     * appendMPSFinalize. This is a workaround to avoid the issue of the
     * cutensornet library not allowing multiple calls of appendMPSFinalize.
     *
     * This function either appends a new `Identity` gate to the graph when the
     * gate cache is empty or update the existing gate operator by itself.
     */
    void dummy_tensor_update() {
        if (identiy_gate_ids_.empty()) {
            applyOperation("Identity", {0}, false);
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateUpdateTensorOperator(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* int64_t tensorId*/
            static_cast<int64_t>(identiy_gate_ids_.front()),
            /* void* */
            static_cast<void *>(
                gate_cache_->get_gate_device_ptr(identiy_gate_ids_.front())),
            /* int32_t unitary*/ 1));
    }

    /**
     * @brief Save quantumState information to data provided by a user
     *
     * @param tensorPtr Pointer to tensors provided by a user
     */
    void computeState(int64_t **extentsPtr, void **tensorPtr) {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(getTNCudaHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* std::size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused as of v24.03*/ 0x0));

        std::size_t worksize =
            getWorkSpaceMemorySize(getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t);
        DataBuffer<std::size_t, int> d_scratch(d_scratch_length, getDevTag(),
                                               true);

        setWorkSpaceMemory(getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ getTNCudaHandle(),
            /* cutensornetState_t */ getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr,
            /* int64_t *stridesOut */ nullptr,
            /* void * */ tensorPtr,
            /* cudaStream_t */ getDevTag().getStreamID()));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
